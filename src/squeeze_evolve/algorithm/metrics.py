"""Data definitions: problem state and per-loop metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Problem state
# ---------------------------------------------------------------------------

@dataclass
class ProblemState:
    orig_prompt: Any  # str for text-only, MultimodalPrompt for vision benchmarks
    gt: Any = None
    candidates: Optional[list[str]] = None
    candidate_groups: Optional[list[list[str]]] = None
    routing_details: Optional[dict[str, Any]] = None
    question: Optional[str] = None   # raw question text for judge prompts
    options: Optional[Any] = None    # answer options for judge prompts (MMMU Pro)


# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoutingMetrics:
    model_counts: dict[int, int] = field(default_factory=dict)
    lite_count: int = 0
    median_thresholds: list[float] = field(default_factory=list)
    per_problem_thresholds: list[list[float]] = field(default_factory=list)


@dataclass
class TokenMetrics:
    model_input_tokens: dict[int, int] = field(default_factory=dict)
    model_output_tokens: dict[int, int] = field(default_factory=dict)
    scoring_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return sum(self.model_input_tokens.values()) + self.scoring_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return sum(self.model_output_tokens.values())


@dataclass
class TimingMetrics:
    time_scoring_s: float = 0.0
    time_generation_s: float = 0.0
    time_total_s: float = 0.0


@dataclass
class ConfidenceMetrics:
    num_candidates_scored: int = 0
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    std_confidence: float = 0.0

    @classmethod
    def from_scores(cls, scores: list[float]) -> ConfidenceMetrics:
        if not scores:
            return cls()
        return cls(
            num_candidates_scored=len(scores),
            mean_confidence=round(float(np.mean(scores)), 4),
            median_confidence=round(float(np.median(scores)), 4),
            min_confidence=round(float(np.min(scores)), 4),
            max_confidence=round(float(np.max(scores)), 4),
            std_confidence=round(float(np.std(scores)), 4),
        )


@dataclass
class LoopMetrics:
    loop: int
    routing: RoutingMetrics = field(default_factory=RoutingMetrics)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    confidence: ConfidenceMetrics = field(default_factory=ConfidenceMetrics)
    eval: dict[str, Any] = field(default_factory=dict)
    confidence_percentiles: list[float] = field(default_factory=lambda: [50.0])
    fitness: str = "confidence"

    def to_flat_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"loop": self.loop}
        # Routing
        for tier_idx, count in sorted(self.routing.model_counts.items()):
            d[f"model_{tier_idx}_count"] = count
        d["lite_count"] = self.routing.lite_count
        d["median_thresholds"] = self.routing.median_thresholds
        d["per_problem_thresholds"] = self.routing.per_problem_thresholds
        # Tokens
        for tier_idx, tok in sorted(self.tokens.model_input_tokens.items()):
            d[f"model_{tier_idx}_input_tokens"] = tok
        for tier_idx, tok in sorted(self.tokens.model_output_tokens.items()):
            d[f"model_{tier_idx}_output_tokens"] = tok
        d["scoring_input_tokens"] = self.tokens.scoring_input_tokens
        d["total_input_tokens"] = self.tokens.total_input_tokens
        d["total_output_tokens"] = self.tokens.total_output_tokens
        # Timing
        d["time_scoring_s"] = self.timing.time_scoring_s
        d["time_generation_s"] = self.timing.time_generation_s
        d["time_total_s"] = self.timing.time_total_s
        # Confidence
        d["num_candidates_scored"] = self.confidence.num_candidates_scored
        d["mean_confidence"] = self.confidence.mean_confidence
        d["median_confidence"] = self.confidence.median_confidence
        d["min_confidence"] = self.confidence.min_confidence
        d["max_confidence"] = self.confidence.max_confidence
        d["std_confidence"] = self.confidence.std_confidence
        # Eval
        for k, v in self.eval.items():
            d[f"eval_{k}"] = v
        d["confidence_percentiles"] = self.confidence_percentiles
        d["fitness"] = self.fitness
        return d
