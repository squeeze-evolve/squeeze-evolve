"""Verify metrics dataclasses and serialization."""

from squeeze_evolve.algorithm.metrics import (
    ConfidenceMetrics,
    LoopMetrics,
    RoutingMetrics,
    TimingMetrics,
    TokenMetrics,
)
from squeeze_evolve.algorithm.utils import aggregate_eval_results


def test_token_metrics_totals() -> None:
    t = TokenMetrics(
        model_input_tokens={0: 100, 1: 200},
        model_output_tokens={0: 300, 1: 400},
        scoring_input_tokens=50,
    )
    assert t.total_input_tokens == 350
    assert t.total_output_tokens == 700


def test_confidence_metrics_from_scores() -> None:
    c = ConfidenceMetrics.from_scores([1.0, 2.0, 3.0, 4.0, 5.0])
    assert c.num_candidates_scored == 5
    assert c.mean_confidence == 3.0
    assert c.min_confidence == 1.0
    assert c.max_confidence == 5.0


def test_confidence_metrics_empty() -> None:
    c = ConfidenceMetrics.from_scores([])
    assert c.num_candidates_scored == 0
    assert c.mean_confidence == 0.0


def test_loop_metrics_flat_dict_has_core_keys() -> None:
    m = LoopMetrics(loop=0)
    flat = m.to_flat_dict()
    core_keys = {
        "loop", "lite_count", "median_thresholds", "per_problem_thresholds",
        "scoring_input_tokens", "total_input_tokens", "total_output_tokens",
        "time_scoring_s", "time_generation_s", "time_total_s",
        "num_candidates_scored",
        "mean_confidence", "median_confidence",
        "min_confidence", "max_confidence", "std_confidence",
        "confidence_percentiles", "fitness",
    }
    assert core_keys.issubset(flat.keys())


def test_loop0_flat_dict() -> None:
    m = LoopMetrics(
        loop=0,
        routing=RoutingMetrics(model_counts={1: 32}),
        tokens=TokenMetrics(model_input_tokens={1: 1000}, model_output_tokens={1: 5000}),
        timing=TimingMetrics(time_generation_s=1.5, time_total_s=1.5),
    )
    flat = m.to_flat_dict()
    assert flat["loop"] == 0
    assert flat["model_1_count"] == 32
    assert flat["model_1_input_tokens"] == 1000
    assert flat["model_1_output_tokens"] == 5000
    assert flat["total_input_tokens"] == 1000
    assert flat["total_output_tokens"] == 5000


def test_evolve_loop_flat_dict() -> None:
    m = LoopMetrics(
        loop=1,
        routing=RoutingMetrics(model_counts={0: 8, 1: 8}, median_thresholds=[2.5], per_problem_thresholds=[[2.5]]),
        tokens=TokenMetrics(
            model_input_tokens={0: 500, 1: 500},
            model_output_tokens={0: 2000, 1: 3000},
            scoring_input_tokens=12000,
        ),
        timing=TimingMetrics(time_scoring_s=0.8, time_generation_s=2.1, time_total_s=2.9),
        confidence=ConfidenceMetrics.from_scores([1.0, 2.0, 3.0, 4.0]),
    )
    flat = m.to_flat_dict()
    assert flat["total_input_tokens"] == 500 + 500 + 12000
    assert flat["total_output_tokens"] == 2000 + 3000
    assert flat["num_candidates_scored"] == 4
    assert flat["model_0_count"] == 8
    assert flat["model_1_count"] == 8
    assert flat["median_thresholds"] == [2.5]
    assert flat["per_problem_thresholds"] == [[2.5]]


def test_three_model_flat_dict() -> None:
    m = LoopMetrics(
        loop=1,
        routing=RoutingMetrics(model_counts={0: 4, 1: 6, 2: 6}, lite_count=0, median_thresholds=[2.0, 4.0], per_problem_thresholds=[[2.0, 4.0]]),
        tokens=TokenMetrics(
            model_input_tokens={0: 100, 1: 200, 2: 300},
            model_output_tokens={0: 400, 1: 500, 2: 600},
            scoring_input_tokens=50,
        ),
    )
    flat = m.to_flat_dict()
    assert flat["model_0_count"] == 4
    assert flat["model_1_count"] == 6
    assert flat["model_2_count"] == 6
    assert flat["model_0_input_tokens"] == 100
    assert flat["model_2_output_tokens"] == 600
    assert flat["total_input_tokens"] == 100 + 200 + 300 + 50
    assert flat["total_output_tokens"] == 400 + 500 + 600
    assert flat["median_thresholds"] == [2.0, 4.0]
    assert flat["per_problem_thresholds"] == [[2.0, 4.0]]


# --- Flexible eval metrics ---

def testaggregate_eval_results_math() -> None:
    results = [
        {"mean_acc": 0.75, "pass_at_k": 1.0, "majority_vote": 1.0},
        {"mean_acc": 0.25, "pass_at_k": 1.0, "majority_vote": 0.0},
        {"mean_acc": 0.0, "pass_at_k": 0.0, "majority_vote": 0.0},
    ]
    agg = aggregate_eval_results(results)
    assert agg["n_evaluated"] == 3
    assert abs(agg["pass_at_k"] - 0.6667) < 0.01
    assert abs(agg["majority_vote"] - 0.3333) < 0.01


def testaggregate_eval_results_code() -> None:
    results = [
        {"pass_at_1": 1.0, "compile_rate": 1.0},
        {"pass_at_1": 0.0, "compile_rate": 0.5},
    ]
    agg = aggregate_eval_results(results)
    assert agg["n_evaluated"] == 2
    assert agg["pass_at_1"] == 0.5
    assert agg["compile_rate"] == 0.75


def testaggregate_eval_results_empty() -> None:
    agg = aggregate_eval_results([])
    assert agg["n_evaluated"] == 0


def test_aggregate_skips_pred_accuracies() -> None:
    results = [{"mean_acc": 1.0, "pred_accuracies": [1.0, 0.0]}]
    agg = aggregate_eval_results(results)
    assert "pred_accuracies" not in agg
    assert agg["mean_acc"] == 1.0


def test_eval_in_flat_dict_prefixed() -> None:
    m = LoopMetrics(
        loop=0,
        eval={"mean_acc": 0.8, "pass_at_k": 1.0, "n_evaluated": 10},
    )
    flat = m.to_flat_dict()
    assert flat["eval_mean_acc"] == 0.8
    assert flat["eval_pass_at_k"] == 1.0
    assert flat["eval_n_evaluated"] == 10
    assert "mean_acc" not in flat  # no collision with un-prefixed keys


def test_eval_custom_keys_in_flat_dict() -> None:
    m = LoopMetrics(
        loop=1,
        eval={"pass_at_1": 0.9, "compile_rate": 0.95, "n_evaluated": 5},
    )
    flat = m.to_flat_dict()
    assert flat["eval_pass_at_1"] == 0.9
    assert flat["eval_compile_rate"] == 0.95
