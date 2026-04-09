import random

import numpy as np
import pytest

from squeeze_evolve.algorithm.operators import (
    assign_routes,
    compute_thresholds,
    evaluation,
    fitness,
    group_confidence,
    group_diversity,
    lite_agg,
    lite_aggregate_majority,
    lite_aggregate_random,
    recombination,
    select_uniform,
    select_weighted,
    selection,
    update,
    update_accumulate,
    update_replace,
)
from squeeze_evolve.common import (
    make_aggregate_prompt,
    strip_think_blocks,
    synthesize_prompt,
)
from squeeze_evolve.common import extract_boxed_math_answer
from squeeze_evolve.core.registry import Registry


# --- Registry ---

def test_registry_register_and_get() -> None:
    r: Registry = Registry("test")

    @r.register("foo")
    def foo() -> str:
        return "bar"

    assert r.get("foo")() == "bar"


def test_registry_unknown_key_raises() -> None:
    r: Registry = Registry("test")
    with pytest.raises(KeyError, match="Unknown test operator"):
        r.get("nonexistent")


def test_registry_duplicate_raises() -> None:
    r: Registry = Registry("test")

    @r.register("dup")
    def first() -> None:
        pass

    with pytest.raises(KeyError, match="already registered"):

        @r.register("dup")
        def second() -> None:
            pass


def test_registry_contains_and_keys() -> None:
    r: Registry = Registry("test")

    @r.register("a")
    def a() -> None:
        pass

    assert "a" in r
    assert "b" not in r
    assert r.keys() == ["a"]


# --- built-in registries ---

def test_builtin_fitness_registry() -> None:
    assert "confidence" in fitness
    assert "diversity" in fitness


def test_builtin_selection_registry() -> None:
    assert "uniform" in selection
    assert "weighted" in selection


def test_builtin_lite_agg_registry() -> None:
    assert "majority" in lite_agg
    assert "random" in lite_agg


def test_builtin_update_registry() -> None:
    assert "replace" in update
    assert "accumulate" in update


def test_benchmark_recombination_registry() -> None:
    assert "aime25-aggregate" in recombination
    assert "gpqa_diamond-aggregate" in recombination


# --- strip_think_blocks ---

def test_strip_think_blocks_removes_tags() -> None:
    assert strip_think_blocks("<think>internal</think>answer") == "answer"


def test_strip_think_blocks_removes_orphan_close() -> None:
    assert strip_think_blocks("some preamble</think>answer") == "answer"


def test_strip_think_blocks_passthrough() -> None:
    assert strip_think_blocks("no tags here") == "no tags here"


# --- fitness ---

def test_group_confidence_mean() -> None:
    assert group_confidence([1.0, 3.0]) == 2.0


def test_group_diversity_counts_unique() -> None:
    assert group_diversity(["a", "b", "a"]) == 2.0


# --- selection ---

def test_select_uniform_shapes() -> None:
    random.seed(42)
    groups, indices = select_uniform(["a", "b", "c", "d", "e"], k=2, m=3)
    assert len(groups) == 3
    assert all(len(g) == 2 for g in groups)


def test_select_weighted_respects_temperature() -> None:
    random.seed(42)
    np.random.seed(42)
    groups, _ = select_weighted(
        ["a", "b", "c", "d", "e"], k=1, m=10,
        scores=[0.0, 0.0, 0.0, 0.0, 100.0], temperature=0.01,
    )
    assert sum(1 for g in groups if "e" in g) >= 8


# --- routing ---

def test_compute_thresholds_single() -> None:
    thresholds = compute_thresholds([1.0, 2.0, 3.0, 4.0], [50.0])
    assert len(thresholds) == 1
    assert abs(thresholds[0] - 2.5) < 0.01


def test_compute_thresholds_multiple() -> None:
    thresholds = compute_thresholds([1.0, 2.0, 3.0, 4.0, 5.0], [25.0, 75.0])
    assert len(thresholds) == 2
    assert thresholds[0] < thresholds[1]


def test_compute_thresholds_extremes() -> None:
    thresholds = compute_thresholds([1.0, 2.0], [0.0, 100.0])
    assert thresholds[0] < 1.0
    assert thresholds[1] > 2.0


def test_assign_routes_two_tier() -> None:
    routes = assign_routes([3.0, 1.0, 4.0, 2.0], [2.5], n_models=2)
    assert routes == ["model_0", "model_1", "model_0", "model_1"]


def test_assign_routes_three_tier_with_lite() -> None:
    routes = assign_routes([3.0, 1.0, 5.0, 2.0], [2.5], n_models=2, lite_fraction=0.5)
    assert routes[2] == "lite"
    assert routes[0] == "model_0"
    assert routes[1] == "model_1"


def test_assign_routes_three_models() -> None:
    # 3 models, 2 thresholds at 2.0 and 4.0
    # fitness 1.0 <= 2.0 -> model_2 (most expensive)
    # fitness 3.0: 2.0 < 3.0 <= 4.0 -> model_1 (mid)
    # fitness 5.0 > 4.0 -> model_0 (cheapest)
    routes = assign_routes([1.0, 3.0, 5.0, 2.0, 4.0], [2.0, 4.0], n_models=3)
    assert routes[0] == "model_2"  # 1.0 <= 2.0
    assert routes[1] == "model_1"  # 2.0 < 3.0 <= 4.0
    assert routes[2] == "model_0"  # 5.0 > 4.0
    assert routes[3] == "model_2"  # 2.0 <= 2.0
    assert routes[4] == "model_1"  # 2.0 < 4.0 <= 4.0


def test_assign_routes_three_models_with_lite() -> None:
    routes = assign_routes([1.0, 3.0, 5.0, 6.0], [2.0, 4.0], n_models=3, lite_fraction=0.5)
    # model_0 candidates: indices 2 (5.0) and 3 (6.0)
    # lite carves top 50% of model_0 -> index 3 (highest fitness) becomes lite
    assert routes[3] == "lite"
    assert routes[2] == "model_0"
    assert routes[0] == "model_2"
    assert routes[1] == "model_1"


def test_assign_routes_single_model() -> None:
    # Single model: no thresholds, everything goes to model_0
    routes = assign_routes([1.0, 2.0, 3.0], [], n_models=1)
    assert routes == ["model_0", "model_0", "model_0"]


# --- lite agg ---

def test_lite_aggregate_majority() -> None:
    assert lite_aggregate_majority(["a", "b", "a"]) == "a"


def test_lite_aggregate_random_in_group() -> None:
    random.seed(0)
    assert lite_aggregate_random(["x", "y", "z"]) in ("x", "y", "z")


# --- update ---

def test_update_replace() -> None:
    assert update_replace(["old"], ["new"]) == ["new"]


def test_update_accumulate() -> None:
    assert update_accumulate(["old"], ["new"]) == ["old", "new"]


# --- recombination: aggregate (via make_aggregate_prompt factory) ---

def test_aggregate_no_candidates() -> None:
    agg = make_aggregate_prompt("math problem", "\\boxed{}")
    assert agg("Q?", []) == "Q?"


def test_aggregate_single_candidate_math() -> None:
    agg = make_aggregate_prompt("math problem", "\\boxed{}")
    p = agg("Solve x+1=3", ["x=2"])
    assert "math problem" in p
    assert "---- Candidate ----" in p
    assert "\\boxed{}" in p


def test_aggregate_multi_candidate_math() -> None:
    agg = make_aggregate_prompt("math problem", "\\boxed{}")
    p = agg("Solve x+1=3", ["x=2", "x=3"])
    assert "---- Solution 1 ----" in p
    assert "---- Solution 2 ----" in p


def test_aggregate_code_task() -> None:
    agg = make_aggregate_prompt("code implementation problem", "```python\\n...\\n```", is_code=True)
    p = agg("FizzBuzz", ["def f(): ..."])
    assert "Refine this code" in p


def test_aggregate_discover_task() -> None:
    agg = make_aggregate_prompt("code optimization problem", "```python\\n...\\n```", is_code=True)
    p = agg("Optimize", ["def run(): ..."])
    assert "Refine this code" in p


def test_aggregate_rg_task() -> None:
    agg = make_aggregate_prompt("problem", "<answer>...</answer>")
    assert "<answer>" in agg("Count", ["42", "43"])


def test_aggregate_gpqa_task() -> None:
    agg = make_aggregate_prompt("multiple-choice problem", "\\boxed{}")
    assert "multiple-choice" in agg("Which?", ["A", "B"])


def test_registered_aime25_aggregate() -> None:
    agg = recombination.get("aime25-aggregate")
    p = agg("Solve x=1", ["x=1"])
    assert "math problem" in p




# --- recombination: synthesize ---

def test_synthesize_no_candidates() -> None:
    assert synthesize_prompt("Q?", []) == "Q?"


def test_synthesize_with_candidates() -> None:
    p = synthesize_prompt("Q?", ["A1", "A2"])
    assert "Candidate 1: A1" in p
    assert "Synthesize" in p


# --- evaluation ---

from squeeze_evolve.common import eval_exact_match, eval_none


def test_benchmark_evaluation_registry() -> None:
    assert "aime25-boxed_math" in evaluation
    assert "aime25-exact_match" in evaluation
    assert "aime25-none" in evaluation


def test_eval_exact_match_all_correct() -> None:
    r = eval_exact_match(["42", "42", "42"], "42")
    assert r["mean_acc"] == 1.0
    assert r["pass_at_k"] == 1.0
    assert r["majority_vote"] == 1.0


def test_eval_exact_match_partial() -> None:
    r = eval_exact_match(["42", "43", "42"], "42")
    assert abs(r["mean_acc"] - 2 / 3) < 0.01
    assert r["pass_at_k"] == 1.0
    assert r["majority_vote"] == 1.0  # majority is "42"


def test_eval_exact_match_none_correct() -> None:
    r = eval_exact_match(["43", "44"], "42")
    assert r["mean_acc"] == 0.0
    assert r["pass_at_k"] == 0.0


def test_eval_none_returns_zeros() -> None:
    r = eval_none(["anything"], "gt")
    assert r["mean_acc"] == 0.0
    assert r["pass_at_k"] == 0.0


def test_extract_boxed_math_answer_prefers_boxed() -> None:
    text = "Reasoning...\nFinal Answer: 71\nTherefore the answer is \\boxed{70}."
    assert extract_boxed_math_answer(text) == "70"


def test_extract_boxed_math_answer_falls_back_to_final_answer() -> None:
    text = "Work here\nFinal Answer: 588."
    assert extract_boxed_math_answer(text) == "588"


def test_eval_boxed_math_scores_majority_vote() -> None:
    eval_boxed_math = evaluation.get("aime25-boxed_math")
    r = eval_boxed_math(
        ["After solving, \\boxed{70}", "Final Answer: 70", "\\boxed{71}"],
        "70",
    )
    assert abs(r["mean_acc"] - 2 / 3) < 0.01
    assert r["pass_at_k"] == 1.0
    assert r["majority_vote"] == 1.0
