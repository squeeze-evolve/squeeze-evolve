import asyncio
from unittest.mock import AsyncMock, patch

from squeeze_evolve.algorithm.metrics import ProblemState
from squeeze_evolve.algorithm.orchestrator import RoutingOrchestrator
from squeeze_evolve.core.backend import (
    GenerationResponse,
    OpenAIBackend,
    RetryPolicy,
    _sanitize_openai_prompt,
    run_with_retry,
)
from squeeze_evolve.core.config import ModelConfig, RetryConfig, RoutingConfig, RunConfig


class _TransientError(Exception):
    status_code = 500


def test_backoff_bounds() -> None:
    policy = RetryPolicy(max_retries=3, base_backoff_seconds=1, max_backoff_seconds=2, jitter_seconds=0)
    assert policy.backoff_seconds(0) == 1
    assert policy.backoff_seconds(1) == 2
    assert policy.backoff_seconds(3) == 2


def test_run_with_retry_eventually_succeeds() -> None:
    attempts = {"n": 0}
    policy = RetryPolicy(max_retries=4, base_backoff_seconds=0.001, max_backoff_seconds=0.002, jitter_seconds=0)

    async def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _TransientError("boom")
        return "ok"

    result = asyncio.run(run_with_retry(fn, policy))
    assert result == "ok"
    assert attempts["n"] == 3


def test_generate_batched_splits_prompts() -> None:
    """generate_batched should call generate in chunks of batch_size."""
    cfg = ModelConfig(name="test", base_url="http://localhost:8000/v1")
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    call_sizes: list[int] = []
    original_generate = backend.generate

    async def mock_generate(prompts):
        call_sizes.append(len(prompts))
        return [GenerationResponse(text=f"r{i}") for i in range(len(prompts))]

    backend.generate = mock_generate  # type: ignore[assignment]

    results = asyncio.run(backend.generate_batched(["p"] * 7, batch_size=3))
    assert len(results) == 7
    assert call_sizes == [3, 3, 1]


def test_generate_omits_reasoning_effort_when_unset() -> None:
    cfg = ModelConfig(name="gpt-5-mini", base_url="https://api.openai.com/v1")
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    choice = type("Choice", (), {"message": type("Msg", (), {"content": "ok"})()})()
    response = type("Response", (), {"choices": [choice], "usage": None})()
    backend.client.chat.completions.create = AsyncMock(return_value=response)

    result = asyncio.run(backend.generate(["hello"]))

    assert result[0].text == "ok"
    backend.client.chat.completions.create.assert_awaited_once()
    assert "reasoning_effort" not in backend.client.chat.completions.create.await_args.kwargs


def test_generate_passes_reasoning_effort_when_set() -> None:
    cfg = ModelConfig(
        name="gpt-5-mini",
        base_url="https://api.openai.com/v1",
        reasoning_effort="medium",
    )
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    choice = type("Choice", (), {"message": type("Msg", (), {"content": "ok"})()})()
    response = type("Response", (), {"choices": [choice], "usage": None})()
    backend.client.chat.completions.create = AsyncMock(return_value=response)

    result = asyncio.run(backend.generate(["hello"]))

    assert result[0].text == "ok"
    backend.client.chat.completions.create.assert_awaited_once()
    assert backend.client.chat.completions.create.await_args.kwargs["reasoning_effort"] == "medium"


def test_prompt_confidence_sends_vllm_fields_via_extra_body() -> None:
    cfg = ModelConfig(
        name="scorer",
        base_url="http://localhost:8000/v1",
        prompt_logprobs=True,
        vllm_extensions=True,
    )
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    choice = type("Choice", (), {"mean_prompt_confidence": 0.42, "prompt_logprobs": []})()
    response = type("Response", (), {"choices": [choice]})()
    backend.client.completions.create = AsyncMock(return_value=response)

    scores = asyncio.run(backend.prompt_confidence(["prompt + answer"], [7]))

    assert scores == [0.42]
    backend.client.completions.create.assert_awaited_once_with(
        model="scorer",
        prompt="prompt + answer",
        max_tokens=1,
        temperature=0,
        extra_body={
            "prompt_logprobs": 5,
            "prompt_confidence_only": True,
            "confidence_start_idx": [7],
        },
    )


def test_count_tokens_uses_vllm_tokenize_endpoint() -> None:
    cfg = ModelConfig(
        name="scorer",
        base_url="http://localhost:8000/v1",
        prompt_logprobs=True,
        vllm_extensions=True,
    )
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    backend._post_json = lambda url, payload: {  # type: ignore[method-assign]
        "count": 5 if payload["prompt"] == "alpha" else 9,
    }

    counts = asyncio.run(backend.count_tokens(["alpha", "beta gamma"]))

    assert counts == [5, 9]


def test_sanitize_openai_prompt_strips_tikz_block() -> None:
    prompt = "Geometry\n\\begin{tikzpicture}\nline\n\\end{tikzpicture}\nAnswer."
    sanitized = _sanitize_openai_prompt(prompt)
    assert "\\begin{tikzpicture}" not in sanitized
    assert "[diagram omitted]" in sanitized


def test_score_population_uses_group_context_and_chat_template(monkeypatch, tmp_path) -> None:
    class DummyStorage:
        def save_json(self, key, data):  # noqa: ANN001, D401
            return None

        def load_json(self, key):  # noqa: ANN001, D401
            raise FileNotFoundError(key)

        def list_files(self, prefix=""):  # noqa: ANN001, D401
            return []

        def exists(self, key):  # noqa: ANN001, D401
            return False

    class FakeBackend:
        def __init__(self, token_counts=None):
            self.token_counts = token_counts or []
            self.last_starts = None

        async def generate(self, prompts):  # noqa: ANN001
            return [GenerationResponse(text="unused") for _ in prompts]

        async def generate_batched(self, prompts, batch_size=32):  # noqa: ANN001
            return [GenerationResponse(text="unused") for _ in prompts]

        async def count_tokens(self, texts):  # noqa: ANN001
            return self.token_counts

        async def prompt_confidence(self, full_texts, start_idxs):  # noqa: ANN001
            self.last_starts = list(start_idxs)
            return [0.9 for _ in full_texts]

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ANN001
            content = messages[0]["content"]
            return f"<|user|>\n{content}<|end|>\n<|assistant|>\n"

        def encode(self, text, add_special_tokens=True):  # noqa: ANN001
            return list(range(len(text.split())))

        @classmethod
        def from_pretrained(cls, name, trust_remote_code=False):  # noqa: ANN001
            return cls()

    cheap = FakeBackend()
    expensive = FakeBackend()
    scorer = FakeBackend()
    created = [cheap, expensive, scorer]

    def fake_make_backend(model_cfg, retry_cfg):  # noqa: ANN001
        return created.pop(0)

    monkeypatch.setattr("squeeze_evolve.algorithm.orchestrator.make_backend", fake_make_backend)
    monkeypatch.setattr("squeeze_evolve.algorithm.orchestrator.create_storage", lambda path: DummyStorage())

    import types
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    cfg = RunConfig(
        run_name="token-test",
        routing=RoutingConfig(k=1, population=1, groups=1, loops=2, fitness="confidence", task="math", recombination="aime25-aggregate", evaluation="aime25-none"),
        models=[
            ModelConfig(name="m0", base_url="http://localhost:8000/v1"),
            ModelConfig(name="m1", base_url="http://localhost:8001/v1"),
        ],
        scoring_model=ModelConfig(
            name="scorer",
            base_url="http://localhost:8000/v1",
            prompt_logprobs=True,
            vllm_extensions=True,
        ),
        retry=RetryConfig(max_retries=1),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        metrics_path=str(tmp_path / "metrics.json"),
    )

    orchestrator = RoutingOrchestrator(cfg)
    problems = [ProblemState(orig_prompt="Solve", gt="42", candidates=["\\boxed{42}"], candidate_groups=[[]])]

    scores, _ = asyncio.run(orchestrator._score_population(problems))  # noqa: SLF001

    assert scores == {(0, 0): 0.9}
    # start_idx is computed via FakeTokenizer.encode on the chat-template-formatted prompt
    assert scorer.last_starts is not None
    assert len(scorer.last_starts) == 1


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        content = messages[0]["content"]
        return f"<|user|>\n{content}<|end|>\n<|assistant|>\n"

    @classmethod
    def from_pretrained(cls, name, trust_remote_code=False):
        return cls()


def test_generate_completions_endpoint(monkeypatch) -> None:
    """When endpoint='completions', uses client.completions.create and applies chat template."""
    cfg = ModelConfig(name="test-model", base_url="http://localhost:8000/v1", endpoint="completions")
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    # Inject fake tokenizer
    backend._tokenizer = _FakeTokenizer()

    choice = type("Choice", (), {"text": "result"})()
    response = type("Response", (), {"choices": [choice], "usage": None})()
    backend.client.completions.create = AsyncMock(return_value=response)

    result = asyncio.run(backend.generate(["hello"]))

    assert result[0].text == "result"
    backend.client.completions.create.assert_awaited_once()
    call_kwargs = backend.client.completions.create.await_args.kwargs
    assert call_kwargs["prompt"] == "<|user|>\nhello<|end|>\n<|assistant|>\n"
    assert "messages" not in call_kwargs


def test_generate_completions_uses_max_tokens(monkeypatch) -> None:
    """Completions endpoint uses 'max_tokens', not 'max_completion_tokens'."""
    cfg = ModelConfig(name="test-model", base_url="http://localhost:8000/v1", endpoint="completions", max_tokens=4096)
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)
    backend._tokenizer = _FakeTokenizer()

    choice = type("Choice", (), {"text": "ok"})()
    response = type("Response", (), {"choices": [choice], "usage": None})()
    backend.client.completions.create = AsyncMock(return_value=response)

    asyncio.run(backend.generate(["test"]))

    call_kwargs = backend.client.completions.create.await_args.kwargs
    assert call_kwargs["max_tokens"] == 4096
    assert "max_completion_tokens" not in call_kwargs


def test_generate_completions_sanitize_retry() -> None:
    """Sanitization retry path works with completions endpoint."""
    cfg = ModelConfig(name="gpt-5-mini", base_url="https://api.openai.com/v1", endpoint="completions")
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)
    backend._tokenizer = _FakeTokenizer()

    class InvalidPromptError(Exception):
        pass

    good_choice = type("Choice", (), {"text": "ok"})()
    good_response = type("Response", (), {"choices": [good_choice], "usage": None})()

    backend.client.completions.create = AsyncMock(
        side_effect=[
            InvalidPromptError("invalid_prompt: flagged as potentially violating our usage policy"),
            good_response,
        ]
    )

    prompt = "Question\n\\begin{tikzpicture}\nline\n\\end{tikzpicture}\nAnswer"
    result = asyncio.run(backend.generate([prompt]))

    assert result[0].text == "ok"
    calls = backend.client.completions.create.await_args_list
    assert len(calls) == 2
    assert "\\begin{tikzpicture}" in calls[0].kwargs["prompt"]
    assert "[diagram omitted]" in calls[1].kwargs["prompt"]


def test_generate_retries_invalid_prompt_with_sanitized_prompt() -> None:
    cfg = ModelConfig(name="gpt-5-mini", base_url="https://api.openai.com/v1")
    retry = RetryConfig(max_retries=1)
    backend = OpenAIBackend(cfg, retry)

    class InvalidPromptError(Exception):
        pass

    good_choice = type("Choice", (), {"message": type("Msg", (), {"content": "ok"})()})()
    good_response = type("Response", (), {"choices": [good_choice], "usage": None})()

    backend.client.chat.completions.create = AsyncMock(
        side_effect=[
            InvalidPromptError("invalid_prompt: flagged as potentially violating our usage policy"),
            good_response,
        ]
    )

    prompt = "Question\n\\begin{tikzpicture}\nline\n\\end{tikzpicture}\nAnswer"
    result = asyncio.run(backend.generate([prompt]))

    assert result[0].text == "ok"
    calls = backend.client.chat.completions.create.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["messages"][0]["content"] == prompt
    assert "[diagram omitted]" in calls[1].kwargs["messages"][0]["content"]
