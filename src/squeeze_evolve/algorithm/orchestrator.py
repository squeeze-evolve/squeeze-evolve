"""Async orchestration engine implementing Algorithm 1 (Squeeze-Evolve).

Loop 0:  Initialize population P_q^(0) with the most expensive model (models[-1]).
Loop t:  Score -> Select -> GroupFitness -> Route -> Agg(model_0 || ... || model_{N-1} || Lite) -> Update.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from typing import Any, Callable, Optional

import numpy as np
from tqdm.auto import tqdm

from ..core.backend import GenerationResponse, make_backend
from ..core.config import RunConfig, validate_scoring_policy
from ..core.storage import create_storage
from ..core.types import MultimodalPrompt, Prompt
from ..common import extract_boxed_math_answer, strip_think_blocks
from .metrics import (
    ConfidenceMetrics,
    LoopMetrics,
    ProblemState,
    RoutingMetrics,
    TimingMetrics,
    TokenMetrics,
)
from .operators import (
    assign_routes,
    compute_thresholds,
    evaluation,
    fitness,
    lite_agg,
    recombination,
    selection,
    update,
)
from .utils import (
    aggregate_eval_results,
    append_metrics,
    load_latest_checkpoint,
    load_run_config,
    save_checkpoint,
    sum_tokens,
    validate_problems,
)

logger = logging.getLogger("squeeze_evolve")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt_text(prompt: Prompt) -> str:
    """Extract the text component of a Prompt (str or MultimodalPrompt)."""
    if isinstance(prompt, MultimodalPrompt):
        return prompt.text
    return prompt


def _prompt_with_images(text: str, source_prompt: Prompt, include_images: bool) -> Prompt:
    """Rebuild a Prompt from recombination *text* and the original prompt's images.

    When *include_images* is ``True`` and the source is multimodal, the
    returned prompt carries the original images. Otherwise a plain string
    is returned (cheaper and sufficient for text-only recombination loops).
    """
    if include_images and isinstance(source_prompt, MultimodalPrompt) and source_prompt.has_images:
        return MultimodalPrompt(text=text, images=source_prompt.images)
    return text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RoutingOrchestrator:
    def __init__(self, cfg: RunConfig):
        validate_scoring_policy(cfg)
        self.cfg = cfg
        self.run_id = f"{cfg.run_name}_{uuid.uuid4().hex[:8]}"

        # models[0] = cheapest, models[-1] = most expensive
        self.backends = [make_backend(m, cfg.retry) for m in cfg.models]
        self.scorer = make_backend(cfg.scoring_model, cfg.retry) if cfg.scoring_model else None

        n = len(self.backends)
        percs = cfg.routing.confidence_percentiles
        if n > 1 and len(percs) != n - 1 and not cfg.routing.multimodal:
            raise ValueError(
                f"{n} models require {n - 1} confidence_percentiles, got {len(percs)}"
            )
        self._percentiles = percs if n > 1 and len(percs) == n - 1 else []

        self._checkpoint_storage = create_storage(cfg.checkpoint_dir)
        self._metrics_storage = create_storage(os.path.dirname(cfg.metrics_path) or ".")
        self._metrics_key = os.path.basename(cfg.metrics_path)

        self._fitness_fn = fitness.get(cfg.routing.fitness)
        self._select_fn = selection.get(cfg.routing.selection)
        self._update_fn = update.get(cfg.routing.update)
        self._lite_agg_fn = lite_agg.get(cfg.routing.lite_method)
        self._recomb_fn = recombination.get(cfg.routing.recombination)
        self._eval_fn = evaluation.get(cfg.routing.evaluation)
        self._batch_size = cfg.routing.generation_batch_size

        self._is_multimodal = cfg.routing.multimodal
        self._include_images_in_recomb = cfg.routing.include_images_in_recombination

        # Scoring tokenizer — needed for chat template + token boundary computation.
        # Skipped for multimodal benchmarks (no fitness-based routing).
        if not self._is_multimodal and cfg.routing.fitness != "diversity":
            scoring_cfg = cfg.scoring_model if cfg.scoring_model else cfg.models[-1]
            tok_name = scoring_cfg.tokenizer or scoring_cfg.name
            try:
                from transformers import AutoTokenizer
                self._scoring_tokenizer = AutoTokenizer.from_pretrained(
                    tok_name, trust_remote_code=True,
                )
            except ImportError:
                raise ImportError(
                    "Confidence-based scoring requires the 'transformers' package. "
                    "Install it with: pip install squeeze_evolve[scoring]"
                ) from None
        else:
            self._scoring_tokenizer = None

    def _recomb(self, orig_prompt: Prompt, candidates: list[str]) -> Prompt:
        """Run the recombination operator and re-attach images if needed."""
        text = _prompt_text(orig_prompt)
        recomb_text = self._recomb_fn(text, candidates, **self._operator_ctx)
        return _prompt_with_images(recomb_text, orig_prompt, self._include_images_in_recomb)

    def _apply_chat_template(self, prompt: Prompt) -> tuple[str, int]:
        """Apply chat template, return (formatted_string, token_count)."""
        text = _prompt_text(prompt)
        messages = [{"role": "user", "content": text}]
        formatted = self._scoring_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        token_ids = self._scoring_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        return formatted, len(token_ids)

    def _strip(self, text: str) -> str:
        """Strip think blocks if configured, otherwise return as-is."""
        if self.cfg.routing.strip_think:
            return strip_think_blocks(text)
        return text

    def _extract_answer(self, candidate: str) -> str:
        """Extract a final answer string for diversity fitness."""
        task = self.cfg.routing.task
        if task in ("math", "gpqa_diamond"):
            return extract_boxed_math_answer(candidate)
        # Generic fallback: try boxed extraction, then strip think blocks.
        answer = extract_boxed_math_answer(candidate)
        if answer:
            return answer
        return strip_think_blocks(candidate or "")

    @property
    def _operator_ctx(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "task": self.cfg.routing.task,
            "temperature": self.cfg.routing.selection_temperature,
        }
        if self.cfg.judge_model is not None:
            ctx["judge_model_cfg"] = self.cfg.judge_model.model_dump()
        return ctx

    # -- Loop 0 -------------------------------------------------------------

    async def _loop0(self, problems: list[ProblemState]) -> tuple[list[ProblemState], LoopMetrics]:
        t0 = time.time()
        N = self.cfg.routing.population
        prompts: list[Prompt] = [self._recomb(p.orig_prompt, []) for p in problems for _ in range(N)]
        top = len(self.backends) - 1
        logger.info("Loop 0: generating %d candidates (%d problems x %d population)", len(prompts), len(problems), N)
        responses = await self.backends[-1].generate_batched(prompts, self._batch_size)
        m_in, m_out = sum_tokens(responses)
        for i, p in enumerate(problems):
            p.candidates = [r.text for r in responses[i * N : (i + 1) * N]]
            p.candidate_groups = [[] for _ in range(N)]
        elapsed = round(time.time() - t0, 3)
        logger.info("Loop 0: done in %.1fs, %d input tokens, %d output tokens", elapsed, m_in, m_out)
        return problems, LoopMetrics(
            loop=0,
            routing=RoutingMetrics(model_counts={top: len(prompts)}),
            tokens=TokenMetrics(model_input_tokens={top: m_in}, model_output_tokens={top: m_out}),
            timing=TimingMetrics(time_generation_s=elapsed, time_total_s=elapsed),
            confidence_percentiles=self.cfg.routing.confidence_percentiles,
            fitness=self.cfg.routing.fitness,
        )

    # -- Score population ----------------------------------------------------

    async def _score_population(self, problems: list[ProblemState]) -> tuple[dict[tuple[int, int], float], int]:
        # Multimodal benchmarks do not support fitness-based routing.
        if self._is_multimodal:
            confs: dict[tuple[int, int], float] = {}
            for p_idx, problem in enumerate(problems):
                if not problem.candidates:
                    continue
                for c_idx in range(len(problem.candidates)):
                    confs[(p_idx, c_idx)] = 0.0
            return confs, 0

        confs = {}
        full_texts, starts, lookup = [], [], []
        for p_idx, problem in enumerate(problems):
            if not problem.candidates:
                continue
            for c_idx, cand in enumerate(problem.candidates):
                stripped = self._strip(cand)
                group = problem.candidate_groups[c_idx]
                recomb_prompt = self._recomb(problem.orig_prompt, group)
                formatted, start_idx = self._apply_chat_template(recomb_prompt)
                full_texts.append(f"{formatted}{stripped}")
                starts.append(start_idx)
                lookup.append((p_idx, c_idx))
        if not full_texts:
            return confs, 0
        scoring_input_tokens = sum(
            len(self._scoring_tokenizer.encode(t, add_special_tokens=False))
            for t in full_texts
        )
        logger.debug("Scoring %d candidates (%d tokens total)", len(full_texts), scoring_input_tokens)
        if self.cfg.model_count == 1 and self.cfg.models[-1].prompt_logprobs:
            scores = await self.backends[-1].prompt_confidence(full_texts, starts)
        else:
            assert self.scorer is not None
            scores = await self.scorer.prompt_confidence(full_texts, starts)
        for key, score in zip(lookup, scores):
            confs[key] = float(score or 0.0)
        return confs, scoring_input_tokens

    # -- Evolve loop (1..T) --------------------------------------------------

    async def _evolve_loop(self, problems: list[ProblemState], t: int) -> tuple[list[ProblemState], LoopMetrics]:
        t0 = time.time()
        rc = self.cfg.routing
        n_backends = len(self.backends)

        # Score
        t_score_start = time.time()
        is_diversity = rc.fitness == "diversity"
        if is_diversity:
            candidate_scores: dict[tuple[int, int], float] = {}
            scoring_input_tokens = 0
        else:
            candidate_scores, scoring_input_tokens = await self._score_population(problems)
        t_score = time.time() - t_score_start

        # Select + Route
        all_routes, all_groups, all_thresholds = [], [], []
        all_group_fitnesses: list[list[float]] = []
        all_indices: list[list[list[int]]] = []
        for q, problem in enumerate(problems):
            assert problem.candidates is not None
            scores_q = [candidate_scores.get((q, c), 0.0) for c in range(len(problem.candidates))]
            groups, indices = self._select_fn(
                problem.candidates, rc.k, rc.groups,
                scores=scores_q, **self._operator_ctx,
            )
            all_groups.append(groups)
            all_indices.append(indices)

            if is_diversity:
                gf = [self._fitness_fn([self._extract_answer(problem.candidates[idx]) for idx in gi]) for gi in indices]
            else:
                gf = [self._fitness_fn([candidate_scores.get((q, idx), 0.0) for idx in gi]) for gi in indices]
            all_group_fitnesses.append(gf)
            thresholds = compute_thresholds(gf, self._percentiles)
            all_routes.append(assign_routes(gf, thresholds, n_backends, rc.lite_fraction))
            all_thresholds.append(thresholds)

        # Flatten by tier
        tier_prompts: dict[str, list[Prompt]] = {f"model_{i}": [] for i in range(n_backends)}
        lite_groups: list[list[str]] = []
        flat_order: list[tuple[str, int]] = []
        flat_groups: list[list[str]] = []
        for q, problem in enumerate(problems):
            for g_idx, group in enumerate(all_groups[q]):
                stripped = [self._strip(x) for x in group]
                route = all_routes[q][g_idx]
                if route == "lite":
                    flat_order.append(("lite", len(lite_groups)))
                    lite_groups.append(stripped)
                else:
                    flat_order.append((route, len(tier_prompts[route])))
                    tier_prompts[route].append(self._recomb(problem.orig_prompt, stripped))
                flat_groups.append(stripped)

        # Parallel aggregation across all model tiers
        t_gen_start = time.time()

        async def _agg_tier(idx: int) -> list[GenerationResponse]:
            key = f"model_{idx}"
            if tier_prompts[key]:
                return await self.backends[idx].generate_batched(tier_prompts[key], self._batch_size)
            return []

        tier_responses = await asyncio.gather(*[_agg_tier(i) for i in range(n_backends)])
        t_gen = time.time() - t_gen_start

        r_lite = [self._lite_agg_fn(g) for g in lite_groups]

        # Token accounting
        model_in: dict[int, int] = {i: 0 for i in range(n_backends)}
        model_out: dict[int, int] = {i: 0 for i in range(n_backends)}
        for i, resps in enumerate(tier_responses):
            inp, out = sum_tokens(resps)
            model_in[i] = inp
            model_out[i] = out

        # Merge in original order
        tiers: dict[str, list[str]] = {}
        for i, resps in enumerate(tier_responses):
            tiers[f"model_{i}"] = [r.text for r in resps]
        tiers["lite"] = r_lite
        cursors: dict[str, int] = {k: 0 for k in tiers}
        merged: list[str] = []
        for tier, _ in flat_order:
            merged.append(tiers[tier][cursors[tier]])
            cursors[tier] += 1

        # Update populations
        M = rc.groups
        cursor = 0
        for q, problem in enumerate(problems):
            new_candidates = merged[cursor : cursor + M]
            new_groups = flat_groups[cursor : cursor + M]
            problem.candidates = self._update_fn(problem.candidates or [], new_candidates)
            problem.candidate_groups = self._update_fn(
                problem.candidate_groups or [], new_groups,
            )
            problem.routing_details = {
                "routes": all_routes[q],
                "thresholds": all_thresholds[q],
                "group_fitnesses": all_group_fitnesses[q],
                "group_index_members": all_indices[q],
                "candidate_confidences": {
                    c: candidate_scores.get((q, c), 0.0)
                    for c in range(len(problem.candidates))
                    if (q, c) in candidate_scores
                },
                "percentiles": rc.confidence_percentiles,
            }
            cursor += M

        flat_routes = [r for routes in all_routes for r in routes]
        model_counts = {i: flat_routes.count(f"model_{i}") for i in range(n_backends)}
        lite_n = flat_routes.count("lite")
        elapsed = round(time.time() - t0, 3)

        parts = " ".join(f"M{i}={model_counts[i]}" for i in range(n_backends))
        logger.info(
            "Loop %d: %s Lite=%d, scoring=%.1fs gen=%.1fs total=%.1fs",
            t, parts, lite_n, t_score, t_gen, elapsed,
        )

        # Median of all per-problem thresholds (each is a list of N-1 values)
        median_thresholds = []
        if all_thresholds and all_thresholds[0]:
            for j in range(len(all_thresholds[0])):
                median_thresholds.append(float(np.median([th[j] for th in all_thresholds])))

        return problems, LoopMetrics(
            loop=t,
            routing=RoutingMetrics(
                model_counts=model_counts,
                lite_count=lite_n,
                median_thresholds=median_thresholds,
                per_problem_thresholds=all_thresholds,
            ),
            tokens=TokenMetrics(
                model_input_tokens=model_in,
                model_output_tokens=model_out,
                scoring_input_tokens=scoring_input_tokens,
            ),
            timing=TimingMetrics(
                time_scoring_s=round(t_score, 3),
                time_generation_s=round(t_gen, 3),
                time_total_s=elapsed,
            ),
            confidence=ConfidenceMetrics.from_scores(list(candidate_scores.values())),
            confidence_percentiles=rc.confidence_percentiles,
            fitness=rc.fitness,
        )

    # -- Evaluation ----------------------------------------------------------

    def _evaluate(self, problems: list[ProblemState]) -> dict[str, Any]:
        results = []
        ctx = self._operator_ctx
        for i, p in enumerate(problems):
            if not p.candidates or p.gt is None:
                continue
            try:
                eval_ctx = dict(ctx)
                if p.question is not None:
                    eval_ctx["question"] = p.question
                if p.options is not None:
                    eval_ctx["options"] = p.options
                results.append(self._eval_fn(p.candidates, p.gt, **eval_ctx))
            except Exception:
                logger.warning("Eval failed for problem %d, skipping", i, exc_info=True)
        agg = aggregate_eval_results(results)
        if agg.get("n_evaluated", 0) > 0:
            logger.info("Eval: %s", {k: v for k, v in agg.items() if k != "n_evaluated"})
        return agg

    # -- Main entry ----------------------------------------------------------

    async def run(
        self,
        problems: list[dict[str, Any]],
        on_loop_complete: Optional[Callable[[LoopMetrics], None]] = None,
    ) -> dict[str, Any]:
        validate_problems(problems)

        if self.cfg.routing.seed is not None:
            random.seed(self.cfg.routing.seed)
            np.random.seed(self.cfg.routing.seed)

        state = [ProblemState(**p) for p in problems]
        if self.cfg.resume:
            last = load_latest_checkpoint(self._checkpoint_storage, self.cfg.run_name)
            if last and "problems" in last:
                state = [ProblemState(**p) for p in last["problems"]]

        logger.info("Starting run %s: %d problems, %d loops", self.run_id, len(state), self.cfg.routing.loops)

        all_metrics: list[LoopMetrics] = []
        loop_iter = tqdm(
            range(self.cfg.routing.loops),
            total=self.cfg.routing.loops,
            desc=f"Loops {self.run_id[:12]}",
            unit="loop",
            dynamic_ncols=True,
        )
        for t in loop_iter:
            if t == 0:
                state, metrics = await self._loop0(state)
            else:
                state, metrics = await self._evolve_loop(state, t)
            metrics.eval = self._evaluate(state)
            all_metrics.append(metrics)
            flat = metrics.to_flat_dict()
            save_checkpoint(self._checkpoint_storage, self.cfg.run_name, t, {"problems": [s.__dict__ for s in state], "metrics": flat})
            append_metrics(self._metrics_storage, self._metrics_key, {"run_id": self.run_id, **flat})
            if on_loop_complete:
                on_loop_complete(metrics)
            loop_iter.set_postfix(loop=t, refresh=False)

        logger.info("Run %s complete: %d loops", self.run_id, len(all_metrics))
        return {"run_id": self.run_id, "metrics": [m.to_flat_dict() for m in all_metrics], "problems": [s.__dict__ for s in state]}
