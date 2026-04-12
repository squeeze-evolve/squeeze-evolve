"""MMMU Pro benchmark operators.

MMMU Pro is a multi-image multiple-choice benchmark with up to 7 images
per question and 10 options (A-J). Evaluation uses LLM-as-judge ONLY
(aligned with lmms-eval-new/logs/eval_batch_samples.py).

Question format (applied during data prep, aligned with lmms-eval-new):
  "{question}\\n{A. opt1\\nB. opt2\\n...}\\n\\nAnswer with the option letter
  from the given choices directly."

All benchmark-specific logic (judge prompts, evaluation functions, judge
backend creation) lives here — no changes to core needed.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import json
import logging
import re
from typing import Any, Callable

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import (
    eval_boxed_math,
    eval_none,
    extract_boxed_math_answer,
    make_aggregate_prompt,
    strip_think_blocks,
    synthesize_prompt,
)

logger = logging.getLogger("squeeze_evolve")

_FMT = "\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}"

# ---------------------------------------------------------------------------
# Judge backend (lazy singleton — created on first judge call)
# ---------------------------------------------------------------------------

_judge_backend = None
_judge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _get_judge_backend(judge_model_cfg: dict[str, Any]) -> Any:
    """Lazily create the judge backend from config dict."""
    global _judge_backend
    if _judge_backend is None:
        from squeeze_evolve.core.backend import make_backend
        from squeeze_evolve.core.config import ModelConfig, RetryConfig

        cfg = ModelConfig(**judge_model_cfg)
        _judge_backend = make_backend(cfg, RetryConfig())
    return _judge_backend


def _judge_call(prompt: str, judge_model_cfg: dict[str, Any]) -> str:
    """Synchronous judge completion call."""
    backend = _get_judge_backend(judge_model_cfg)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        return _judge_executor.submit(
            asyncio.run, backend.judge_completion(prompt),
        ).result()
    return asyncio.run(backend.judge_completion(prompt))


# ---------------------------------------------------------------------------
# LLM-as-judge prompt and evaluation
# Aligned with lmms-eval-new/logs/eval_batch_samples.py
# ---------------------------------------------------------------------------

def _parse_judge_verdict(judge_response: str) -> bool:
    """Parse a True/False verdict from a judge LLM response."""
    text = judge_response.strip().lower()
    return "true" in text


# MMMU Pro judge prompt — aligned with eval_batch_samples.py::build_judge_prompt
MMMU_PRO_JUDGE_PROMPT = """You are an expert evaluator for a multiple-choice exam. Your job is to determine whether the student's response contains the correct answer.

Options:
{options_str}

Correct Answer: {ground_truth}

Student's Response:
{model_response}

Does the student's response indicate the correct answer "{ground_truth}"? The student may have expressed the answer in different ways (e.g., stating the option letter, restating the option content, or arriving at the correct answer through reasoning).

Reply with ONLY "True" or "False"."""


def _eval_mmmu_pro_judge(
    candidates: list[str],
    gt: Any,
    options: list[str] | str | None = None,
    judge_model_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """MMMU Pro evaluation using LLM-as-judge.

    Sends each candidate response along with the options and correct
    answer to the judge model.  Uses LLM judge ONLY — no parser fallback.
    """
    gt_str = str(gt).strip()

    # Parse options
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            try:
                options = json.loads(options)
            except (ValueError, TypeError):
                options = []
    if options is None:
        options = []

    # Format options string: "A. option\nB. option\n..."
    options_str = "\n".join(
        f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
    )

    if judge_model_cfg is None:
        # No judge available — fall back to simple boxed-answer extraction
        extracted = [extract_boxed_math_answer(c) for c in candidates]
        correct = [ans.upper() == gt_str.upper() for ans in extracted]
    else:
        correct = []
        for candidate in candidates:
            # Strip think blocks from response before sending to judge
            model_response = strip_think_blocks(candidate) if candidate else ""
            prompt = MMMU_PRO_JUDGE_PROMPT.format(
                options_str=options_str,
                ground_truth=gt_str,
                model_response=model_response or "(No response)",
            )
            try:
                verdict = _judge_call(prompt, judge_model_cfg)
                correct.append(_parse_judge_verdict(verdict))
            except Exception:
                correct.append(False)

    # Majority vote via judge
    ones = sum(1 for c in correct if c)
    majority_correct = ones > len(correct) / 2 if correct else False

    return {
        "pred_accuracies": [float(c) for c in correct],
        "mean_acc": float(sum(correct)) / max(1, len(correct)),
        "pass_at_k": float(any(correct)),
        "majority_vote": float(majority_correct),
    }


# ---------------------------------------------------------------------------
# Registered operators
# ---------------------------------------------------------------------------

@recombination.register("mmmu_pro-aggregate")
def mmmu_pro_aggregate(query, candidates, **kwargs):
    return make_aggregate_prompt(
        "multiple-choice visual question", _FMT,
    )(query, candidates, **kwargs)


@recombination.register("mmmu_pro-synthesize")
def mmmu_pro_synthesize(query, candidates, **kwargs):
    return synthesize_prompt(query, candidates, **kwargs)


@evaluation.register("mmmu_pro-judge")
def mmmu_pro_judge(candidates, gt, **kwargs):
    """LLM-as-judge evaluation for MMMU Pro.

    Keyword args injected by the orchestrator:
    - ``judge_model_cfg``: dict of ModelConfig fields for the judge
    - ``options``: list of option strings (or JSON-encoded list)
    """
    return _eval_mmmu_pro_judge(
        candidates, gt,
        options=kwargs.get("options"),
        judge_model_cfg=kwargs.get("judge_model_cfg"),
    )


@evaluation.register("mmmu_pro-boxed_math")
def mmmu_pro_boxed_math(candidates, gt, **kwargs):
    """Fallback: boxed-answer extraction without a judge."""
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("mmmu_pro-none")
def mmmu_pro_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
