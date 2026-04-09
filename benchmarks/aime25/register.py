"""AIME 2025 benchmark operators."""

from squeeze_evolve import evaluation, recombination
from squeeze_evolve.common import make_aggregate_prompt, synthesize_prompt
from squeeze_evolve.common import eval_boxed_math, eval_exact_match, eval_none


@recombination.register("aime25-aggregate")
def aime25_aggregate(query, candidates, **kwargs):
    return make_aggregate_prompt("math problem", "\\boxed{}")(query, candidates, **kwargs)


@recombination.register("aime25-synthesize")
def aime25_synthesize(query, candidates, **kwargs):
    return synthesize_prompt(query, candidates, **kwargs)


@evaluation.register("aime25-boxed_math")
def aime25_boxed_math(candidates, gt, **kwargs):
    return eval_boxed_math(candidates, gt, **kwargs)


@evaluation.register("aime25-exact_match")
def aime25_exact_match(candidates, gt, **kwargs):
    return eval_exact_match(candidates, gt, **kwargs)


@evaluation.register("aime25-none")
def aime25_none(candidates, gt, **kwargs):
    return eval_none(candidates, gt, **kwargs)
