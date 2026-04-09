"""Core evolutionary operators mirroring Algorithm 1 from the paper.

Each operator family has a :class:`Registry` instance.  Built-in operators
self-register at import time.  Extend by importing the registry::

    from squeeze_evolve import fitness

    @fitness.register("spread")
    def spread(scores):
        return float(max(scores) - min(scores))
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable, Sequence

import numpy as np

from ..core.registry import Registry

# ---- Registry instances (one per operator family) -------------------------

fitness: Registry[Callable] = Registry("fitness")
selection: Registry[Callable] = Registry("selection")
lite_agg: Registry[Callable] = Registry("lite_agg")
update: Registry[Callable] = Registry("update")
recombination: Registry[Callable] = Registry("recombination")
evaluation: Registry[Callable] = Registry("evaluation")
configs: Registry[Callable] = Registry("configs")


# ---------------------------------------------------------------------------
# Fitness signals  (paper Section 3.2)
# ---------------------------------------------------------------------------

@fitness.register("confidence")
def group_confidence(scores: Sequence[float]) -> float:
    """GC(g) = (1/K) sum_{tau in g} C(tau).  Mean candidate confidence."""
    return float(np.mean(scores))


@fitness.register("diversity")
def group_diversity(answers: Sequence[str]) -> float:
    """D(g) = |{answer(tau) : tau in g}|.  Unique final answers."""
    return float(len(set(answers)))


# ---------------------------------------------------------------------------
# Selection  (paper Section 3.3)
# ---------------------------------------------------------------------------

@selection.register("uniform")
def select_uniform(
    candidates: Sequence[str], k: int, m: int, **kwargs: Any,
) -> tuple[list[list[str]], list[list[int]]]:
    """Uniform random K-subsets.  Returns (string_groups, index_groups)."""
    indices = [random.sample(range(len(candidates)), k) for _ in range(m)]
    groups = [[candidates[i] for i in grp] for grp in indices]
    return groups, indices


@selection.register("weighted")
def select_weighted(
    candidates: Sequence[str], k: int, m: int, **kwargs: Any,
) -> tuple[list[list[str]], list[list[int]]]:
    """Fitness-weighted selection with temperature zeta.

    Keyword args:
        scores: Per-candidate fitness scores (required).
        temperature: Softmax temperature controlling exploitation vs exploration.
    """
    scores: Sequence[float] = kwargs["scores"]
    temperature: float = kwargs.get("temperature", 1.0)
    logits = np.array(scores) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    probs = weights / weights.sum()
    idx_range = list(range(len(candidates)))
    indices = [
        list(np.random.choice(idx_range, size=k, replace=False, p=probs))
        for _ in range(m)
    ]
    groups = [[candidates[i] for i in grp] for grp in indices]
    return groups, indices


# ---------------------------------------------------------------------------
# Routing  (paper Eq. threshold + Eq. routing)
# ---------------------------------------------------------------------------

def compute_thresholds(fitnesses: Sequence[float], percentiles: list[float]) -> list[float]:
    """Compute N-1 thresholds from a percentile list.  Returns sorted ascending."""
    results: list[float] = []
    for p in sorted(percentiles):
        if p >= 100:
            results.append(float(max(fitnesses)) + 1.0)
        elif p <= 0:
            results.append(float(min(fitnesses)) - 1.0)
        else:
            results.append(float(np.percentile(fitnesses, p)))
    return results


def assign_routes(
    fitnesses: Sequence[float],
    thresholds: list[float],
    n_models: int,
    lite_fraction: float = 0.0,
) -> list[str]:
    """Assign each group to ``model_0`` .. ``model_{N-1}`` or ``lite``."""
    routes: list[str] = []
    for f in fitnesses:
        assigned = False
        for i, t in enumerate(thresholds):
            if f <= t:
                routes.append(f"model_{n_models - 1 - i}")
                assigned = True
                break
        if not assigned:
            routes.append("model_0")

    if lite_fraction > 0:
        easy = sorted(
            [i for i, r in enumerate(routes) if r == "model_0"],
            key=lambda i: fitnesses[i], reverse=True,
        )
        for i in easy[: max(1, int(len(easy) * lite_fraction))]:
            routes[i] = "lite"
    return routes


# ---------------------------------------------------------------------------
# LiteAgg  (paper Table 1)
# ---------------------------------------------------------------------------

@lite_agg.register("majority")
def lite_aggregate_majority(group: list[str]) -> str:
    """Majority vote over candidate answers."""
    return Counter(group).most_common(1)[0][0]


@lite_agg.register("random")
def lite_aggregate_random(group: list[str]) -> str:
    """Random pick from the group."""
    return random.choice(group)


# ---------------------------------------------------------------------------
# Population update  (paper Section 3.4)
# ---------------------------------------------------------------------------

@update.register("replace")
def update_replace(old: list[str], new: list[str]) -> list[str]:
    """Replace: P^(t) = R_new."""
    return new


@update.register("accumulate")
def update_accumulate(old: list[str], new: list[str]) -> list[str]:
    """Accumulate: P^(t) = P^(t-1) union R_new."""
    return old + new
