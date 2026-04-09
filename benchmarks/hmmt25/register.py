"""HMMT February 2025 benchmark operators."""

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import make_aggregate_prompt, synthesize_prompt
from squeeze_evolve.common import eval_boxed_math, eval_exact_match, eval_none


@recombination.register("hmmt25-aggregate")
def hmmt25_aggregate(query, candidates, **kwargs):
    return make_aggregate_prompt("math problem", "\\boxed{}")(query, candidates, **kwargs)


@recombination.register("hmmt25-synthesize")
def hmmt25_synthesize(query, candidates, **kwargs):
    return synthesize_prompt(query, candidates, **kwargs)


@evaluation.register("hmmt25-boxed_math")
def hmmt25_boxed_math(candidates, gt, **kwargs):
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("hmmt25-exact_match")
def hmmt25_exact_match(candidates, gt, **kwargs):
    return eval_exact_match(candidates, gt, **kwargs)


@evaluation.register("hmmt25-none")
def hmmt25_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
