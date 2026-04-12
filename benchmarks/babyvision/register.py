"""BabyVision benchmark operators.

BabyVision is a visual question-answering benchmark with single-image
questions. Answers are extracted from ``\\boxed{}`` and verified via
LLM-as-judge.

Post prompt (appended during data prep, aligned with lmms-eval-new):
  "\\nThink about the question and give your final answer in \\boxed{Answer} format."

Judge prompt aligned with lmms-eval-new babyvision/prompt.py.

All benchmark-specific logic (judge prompts, answer extraction, judge
backend creation) lives here — no changes to core needed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from collections import Counter
from typing import Any, Callable

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import (
    eval_boxed_math,
    eval_none,
    make_aggregate_prompt,
    synthesize_prompt,
)

logger = logging.getLogger("squeeze_evolve")

_FMT = "\\boxed{Answer}"

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
# BabyVision answer extraction
# Aligned with lmms-eval-new/lmms_eval/tasks/babyvision/
# ---------------------------------------------------------------------------

def _extract_babyvision_boxed_answer(text: str) -> str | None:
    """Extract \\boxed{} content from BabyVision model output.

    Handles <think>...</think> reasoning format by prioritizing content
    after </think>.  Aligned with lmms-eval-new babyvision/utils.py.
    """
    if not text:
        return None

    # For models with <think>...</think>, prioritize content after </think>
    think_end = text.find("</think>")
    if think_end != -1:
        text_after_think = text[think_end + 8:]
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, text_after_think)
        if matches:
            return matches[-1].strip()

    # Standard \\boxed{} extraction from full text
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Fallback: "answer is X" or "Answer: X" patterns
    answer_patterns = [
        r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*[\"']?([A-Za-z0-9,\s\(\)\-]+)[\"']?",
        r"(?:final answer|Final Answer)\s*(?:is|:)\s*[\"']?([A-Za-z0-9,\s\(\)\-]+)[\"']?",
    ]
    for pat in answer_patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# LLM-as-judge prompt and evaluation
# Aligned with lmms-eval-new babyvision/prompt.py
# ---------------------------------------------------------------------------

BABYVISION_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}"""


def _parse_judge_verdict(judge_response: str) -> bool:
    """Parse a True/False verdict from a judge LLM response."""
    text = judge_response.strip().lower()
    return "true" in text


def _eval_babyvision_judge(
    candidates: list[str],
    gt: Any,
    question: str = "",
    judge_model_cfg: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """BabyVision evaluation using LLM-as-judge.

    Extracts the ``\\boxed{}`` answer from each candidate, then asks the
    judge model whether it matches the ground truth.
    """
    gt_str = str(gt).strip()
    extracted = []
    for c in candidates:
        ans = _extract_babyvision_boxed_answer(c)
        if ans is None:
            ans = c[:500] if c else ""
        extracted.append(ans)

    if judge_model_cfg is None:
        correct = [ans.lower() == gt_str.lower() for ans in extracted]
    else:
        correct = []
        for ans in extracted:
            prompt = BABYVISION_JUDGE_PROMPT.format(
                question=question,
                groundtruth=gt_str,
                modeloutput=ans or "(No answer provided)",
            )
            try:
                verdict = _judge_call(prompt, judge_model_cfg)
                correct.append(_parse_judge_verdict(verdict))
            except Exception:
                correct.append(ans.lower() == gt_str.lower())

    counts = Counter(extracted)
    majority = counts.most_common(1)[0][0] if counts else ""
    majority_correct = False
    if judge_model_cfg and majority:
        try:
            v = _judge_call(BABYVISION_JUDGE_PROMPT.format(
                question=question, groundtruth=gt_str,
                modeloutput=majority or "(No answer provided)",
            ), judge_model_cfg)
            majority_correct = _parse_judge_verdict(v)
        except Exception:
            majority_correct = majority.lower() == gt_str.lower()
    else:
        majority_correct = majority.lower() == gt_str.lower()

    return {
        "pred_accuracies": [float(c) for c in correct],
        "mean_acc": float(sum(correct)) / max(1, len(correct)),
        "pass_at_k": float(any(correct)),
        "majority_vote": float(majority_correct),
    }


# ---------------------------------------------------------------------------
# Registered operators
# ---------------------------------------------------------------------------

@recombination.register("babyvision-aggregate")
def babyvision_aggregate(query, candidates, **kwargs):
    return make_aggregate_prompt(
        "visual question", _FMT,
    )(query, candidates, **kwargs)


@recombination.register("babyvision-synthesize")
def babyvision_synthesize(query, candidates, **kwargs):
    return synthesize_prompt(query, candidates, **kwargs)


@evaluation.register("babyvision-judge")
def babyvision_judge(candidates, gt, **kwargs):
    """LLM-as-judge evaluation for BabyVision.

    Keyword args injected by the orchestrator:
    - ``judge_model_cfg``: dict of ModelConfig fields for the judge
    - ``question``: raw question text (for the judge prompt)
    """
    return _eval_babyvision_judge(
        candidates, gt,
        question=kwargs.get("question", ""),
        judge_model_cfg=kwargs.get("judge_model_cfg"),
    )


@evaluation.register("babyvision-boxed_math")
def babyvision_boxed_math(candidates, gt, **kwargs):
    """Fallback: boxed-answer extraction without a judge."""
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("babyvision-none")
def babyvision_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
