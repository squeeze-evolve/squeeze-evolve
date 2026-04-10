"""Backend interface and unified OpenAI-compatible implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Protocol, TypeVar
from urllib.request import Request, urlopen

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from .config import ModelConfig, RetryConfig
from .types import MultimodalPrompt, Prompt

T = TypeVar("T")
logger = logging.getLogger("squeeze_evolve")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GenerationResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_logprob_score: Optional[float] = None


class GenerationBackend(Protocol):
    supports_prompt_logprobs: bool
    supports_cross_model_confidence: bool

    async def generate(self, prompts: List[str]) -> List[GenerationResponse]: ...
    async def prompt_confidence(self, full_texts: List[str], start_idxs: List[int]) -> List[Optional[float]]: ...
    async def count_tokens(self, texts: List[str]) -> List[int]: ...


# ---------------------------------------------------------------------------
# Multimodal message helpers
# ---------------------------------------------------------------------------

def _build_message_content(prompt: Prompt) -> Any:
    """Build OpenAI message content from a Prompt.

    * ``str`` -> plain string (text-only, fast path).
    * ``MultimodalPrompt`` without images -> plain string.
    * ``MultimodalPrompt`` with images -> ``list[dict]`` in OpenAI
      ``image_url`` format.
    """
    if isinstance(prompt, str):
        return prompt
    if not prompt.has_images:
        return prompt.text
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt.text}]
    for img_url in prompt.images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": img_url, "detail": "auto"},
        })
    return parts


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    max_retries: int = 5
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    jitter_seconds: float = 0.5

    def should_retry(self, error: Exception) -> bool:
        if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError, asyncio.TimeoutError)):
            return True
        status_code = getattr(error, "status_code", None)
        return bool(status_code and status_code >= 500)

    def backoff_seconds(self, attempt: int) -> float:
        delay = min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** attempt))
        return delay + random.uniform(0, self.jitter_seconds)


async def run_with_retry(fn: Callable[[], Awaitable[T]], policy: RetryPolicy) -> T:
    last_error: Exception | None = None
    for attempt in range(policy.max_retries):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == policy.max_retries - 1 or not policy.should_retry(exc):
                raise
            await asyncio.sleep(policy.backoff_seconds(attempt))
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _extract_logprob(item: object) -> Optional[float]:
    if isinstance(item, dict):
        v = item.get("logprob")
        return float(v) if v is not None else None
    v = getattr(item, "logprob", None)
    return float(v) if v is not None else None


def _sanitize_openai_prompt(prompt: str) -> str:
    """Remove bulky latex drawing blocks that can trip provider prompt filters."""
    sanitized = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
        "[diagram omitted]",
        prompt,
        flags=re.DOTALL,
    )
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def _is_invalid_prompt_error(error: Exception) -> bool:
    text = str(error).lower()
    return "invalid_prompt" in text or "flagged as potentially violating our usage policy" in text


class OpenAIBackend:
    """Unified async backend for all OpenAI-compatible providers."""

    def __init__(self, cfg: ModelConfig, retry_cfg: RetryConfig):
        self.cfg = cfg
        self.supports_prompt_logprobs: bool = cfg.prompt_logprobs > 0
        self.supports_cross_model_confidence: bool = cfg.vllm_extensions
        self.client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=retry_cfg.request_timeout_seconds,
        )
        self.retry_policy = RetryPolicy(
            max_retries=retry_cfg.max_retries,
            base_backoff_seconds=retry_cfg.base_backoff_seconds,
            max_backoff_seconds=retry_cfg.max_backoff_seconds,
            jitter_seconds=retry_cfg.jitter_seconds,
        )
        self._semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self._model_name = cfg.served_model_name or cfg.name
        self._tokenizer: Any = None

    def _server_root(self) -> str:
        base_url = self.cfg.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url[:-3]
        return base_url

    def _is_openai_api(self) -> bool:
        return "api.openai.com" in self.cfg.base_url

    def _post_json(self, url: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request) as response:  # noqa: S310
            return json.load(response)

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            tok_name = self.cfg.tokenizer or self.cfg.name
            self._tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        return self._tokenizer

    def _apply_chat_template(self, prompt: str) -> str:
        tokenizer = self._get_tokenizer()
        if self.cfg.reasoning_effort is not None:
            from openai_harmony import (
                Conversation, HarmonyEncodingName, Message,
                ReasoningEffort, Role, SystemContent,
                load_harmony_encoding,
            )
            effort_map = {
                "low": ReasoningEffort.LOW,
                "medium": ReasoningEffort.MEDIUM,
                "high": ReasoningEffort.HIGH,
            }
            effort = effort_map.get(self.cfg.reasoning_effort)
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            convo = Conversation.from_messages([
                Message.from_role_and_content(
                    Role.SYSTEM,
                    SystemContent.new().with_reasoning_effort(effort),
                ),
                Message.from_role_and_content(Role.USER, prompt),
            ])
            prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            return tokenizer.decode(prefill_ids)
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _chat_completion_kwargs(self, prompt: Prompt) -> dict[str, Any]:
        content = _build_message_content(prompt)
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_completion_tokens": self.cfg.max_tokens,
        }
        if self.cfg.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort
        if self.cfg.seed is not None:
            kwargs["seed"] = self.cfg.seed
        return kwargs

    def _completion_kwargs(self, prompt: str) -> dict[str, Any]:
        formatted = self._apply_chat_template(prompt)
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "prompt": formatted,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.seed is not None:
            kwargs["seed"] = self.cfg.seed
        return kwargs

    async def _make_call(self, prompt: Prompt) -> GenerationResponse:
        if self.cfg.endpoint == "completions":
            # Completions endpoint only supports text.
            prompt_text = prompt.text if isinstance(prompt, MultimodalPrompt) else prompt
            resp = await self.client.completions.create(**self._completion_kwargs(prompt_text))
            choice = resp.choices[0]
            text = choice.text or ""
        else:
            resp = await self.client.chat.completions.create(**self._chat_completion_kwargs(prompt))
            choice = resp.choices[0]
            text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    async def _one(self, prompt: Prompt) -> GenerationResponse:
        async def call() -> GenerationResponse:
            async with self._semaphore:
                return await self._make_call(prompt)
        try:
            return await run_with_retry(call, self.retry_policy)
        except Exception as exc:  # noqa: BLE001
            prompt_text = prompt.text if isinstance(prompt, MultimodalPrompt) else prompt
            sanitized_prompt = _sanitize_openai_prompt(prompt_text)
            if (
                self._is_openai_api()
                and _is_invalid_prompt_error(exc)
                and sanitized_prompt != prompt_text
            ):
                logger.warning("Retrying invalid_prompt with sanitized prompt for model %s", self._model_name)

                async def retry_call() -> GenerationResponse:
                    async with self._semaphore:
                        return await self._make_call(sanitized_prompt)

                return await run_with_retry(retry_call, self.retry_policy)
            raise

    async def generate(self, prompts: List[Prompt]) -> List[GenerationResponse]:
        return list(await asyncio.gather(*(self._one(p) for p in prompts)))

    async def generate_batched(self, prompts: List[Prompt], batch_size: int = 32) -> List[GenerationResponse]:
        results: List[GenerationResponse] = []
        for i in range(0, len(prompts), batch_size):
            results.extend(await self.generate(prompts[i : i + batch_size]))
        return results

    async def count_tokens(self, texts: List[str]) -> List[int]:
        if not self.cfg.vllm_extensions:
            return [len(text) for text in texts]

        async def _one_count(text: str) -> int:
            async def call() -> int:
                async with self._semaphore:
                    payload = {
                        "model": self._model_name,
                        "prompt": text,
                        "add_special_tokens": True,
                    }
                    data = await asyncio.to_thread(self._post_json, f"{self._server_root()}/tokenize", payload)
                return int(data["count"])

            return await run_with_retry(call, self.retry_policy)

        return list(await asyncio.gather(*(_one_count(text) for text in texts)))

    async def prompt_confidence(self, full_texts: List[str], start_idxs: List[int]) -> List[Optional[float]]:
        if not self.supports_prompt_logprobs:
            return [None for _ in full_texts]

        async def _one_confidence(text: str, start_idx: int) -> Optional[float]:
            async def call() -> Optional[float]:
                async with self._semaphore:
                    extra_body: dict = {"prompt_logprobs": self.cfg.prompt_logprobs}
                    if self.cfg.vllm_extensions:
                        extra_body.update({
                            "prompt_confidence_only": True,
                            "confidence_start_idx": [start_idx],
                        })
                    resp = await self.client.completions.create(
                        model=self._model_name,
                        prompt=text,
                        max_tokens=1,
                        temperature=0,
                        extra_body=extra_body,
                    )
                choice = resp.choices[0]
                if self.cfg.vllm_extensions:
                    scalar = getattr(choice, "mean_prompt_confidence", None)
                    if scalar is not None:
                        return float(scalar)
                prompt_lps = getattr(choice, "prompt_logprobs", None) or []
                vals = [
                    lp for tok in prompt_lps[start_idx:] if tok
                    for val in tok.values()
                    for lp in [_extract_logprob(val)] if lp is not None
                ]
                return float(-sum(vals) / len(vals)) if vals else None
            return await run_with_retry(call, self.retry_policy)

        return list(await asyncio.gather(
            *(_one_confidence(t, s) for t, s in zip(full_texts, start_idxs))
        ))

    # -- Judge completions (text-only, for LLM-as-judge evaluation) ---------

    async def judge_completion(self, prompt: str, **kwargs: Any) -> str:
        """Single text-only judge call. Returns the model's text response.

        This is a simplified call path for LLM-as-judge evaluation
        (e.g. GPT-4o judging BabyVision / MMMU Pro answers).
        """
        async def call() -> str:
            async with self._semaphore:
                resp = await self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_completion_tokens=self.cfg.max_tokens,
                )
                return resp.choices[0].message.content or ""

        return await run_with_retry(call, self.retry_policy)


def make_backend(model_cfg: ModelConfig, retry_cfg: RetryConfig) -> OpenAIBackend:
    return OpenAIBackend(model_cfg, retry_cfg)
