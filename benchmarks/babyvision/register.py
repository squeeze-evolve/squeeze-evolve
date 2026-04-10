"""BabyVision benchmark operators.

BabyVision is a visual question-answering benchmark with single-image
questions. Answers are extracted from ``\\boxed{}`` and verified via
LLM-as-judge.

Post prompt (appended during data prep, aligned with lmms-eval-new):
  "\\nThink about the question and give your final answer in \\boxed{Answer} format."

Judge prompt aligned with lmms-eval-new babyvision/prompt.py.
"""

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import (
    eval_babyvision_judge,
    eval_boxed_math,
    eval_none,
    make_aggregate_prompt,
    synthesize_prompt,
)

_FMT = "\\boxed{Answer}"


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
    - ``judge_fn``: synchronous callable for the judge model
    - ``question``: raw question text (for the judge prompt)
    """
    return eval_babyvision_judge(
        candidates, gt,
        question=kwargs.get("question", ""),
        judge_fn=kwargs.get("judge_fn"),
    )


@evaluation.register("babyvision-boxed_math")
def babyvision_boxed_math(candidates, gt, **kwargs):
    """Fallback: boxed-answer extraction without a judge."""
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("babyvision-none")
def babyvision_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
