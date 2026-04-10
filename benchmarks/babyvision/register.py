"""BabyVision benchmark operators.

BabyVision is a visual question-answering benchmark with single-image
questions. Answers are extracted from ``\\boxed{}`` and verified via
LLM-as-judge (GPT-4o).
"""

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import (
    eval_babyvision_judge,
    eval_boxed_math,
    eval_none,
    make_aggregate_prompt,
    synthesize_prompt,
)

_FMT = "\\boxed{}"


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

    The ``judge_fn`` keyword argument should be a synchronous callable
    that sends a prompt to the judge model and returns the response text.
    It is injected by the orchestrator when a ``judge_model`` is configured.
    """
    return eval_babyvision_judge(
        candidates, gt,
        judge_fn=kwargs.get("judge_fn"),
        **{k: v for k, v in kwargs.items() if k != "judge_fn"},
    )


@evaluation.register("babyvision-boxed_math")
def babyvision_boxed_math(candidates, gt, **kwargs):
    """Fallback: boxed-answer extraction without a judge."""
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("babyvision-none")
def babyvision_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
