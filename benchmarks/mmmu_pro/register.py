"""MMMU Pro benchmark operators.

MMMU Pro is a multi-image multiple-choice benchmark with up to 7 images
per question and 10 options (A-J). Evaluation uses LLM-as-judge (GPT-4o)
— no parser fallback.
"""

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import (
    eval_mmmu_pro_judge,
    eval_boxed_math,
    eval_none,
    make_aggregate_prompt,
    synthesize_prompt,
)

_FMT = "\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}"


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

    The ``judge_fn`` and ``options`` keyword arguments are injected by the
    orchestrator.
    """
    return eval_mmmu_pro_judge(
        candidates, gt,
        options=kwargs.get("options"),
        judge_fn=kwargs.get("judge_fn"),
        **{k: v for k, v in kwargs.items() if k not in ("judge_fn", "options")},
    )


@evaluation.register("mmmu_pro-boxed_math")
def mmmu_pro_boxed_math(candidates, gt, **kwargs):
    """Fallback: boxed-answer extraction without a judge."""
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("mmmu_pro-none")
def mmmu_pro_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
