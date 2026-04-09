"""GPQA-Diamond benchmark operators."""

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import make_aggregate_prompt, synthesize_prompt
from squeeze_evolve.common import eval_boxed_math, eval_exact_match, eval_none

_FMT = "\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}"


@recombination.register("gpqa_diamond-aggregate")
def gpqa_diamond_aggregate(query, candidates, **kwargs):
    return make_aggregate_prompt("multiple-choice problem", _FMT)(query, candidates, **kwargs)


@recombination.register("gpqa_diamond-synthesize")
def gpqa_diamond_synthesize(query, candidates, **kwargs):
    return synthesize_prompt(query, candidates, **kwargs)


@evaluation.register("gpqa_diamond-boxed_math")
def gpqa_diamond_boxed_math(candidates, gt, **kwargs):
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("gpqa_diamond-exact_match")
def gpqa_diamond_exact_match(candidates, gt, **kwargs):
    return eval_exact_match(candidates, gt, **kwargs)


@evaluation.register("gpqa_diamond-none")
def gpqa_diamond_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
