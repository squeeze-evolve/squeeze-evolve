"""Utility helpers for the orchestration engine."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.backend import GenerationResponse
from ..core.config import RunConfig
from .operators import configs


# ---------------------------------------------------------------------------
# Checkpoint helpers (storage-backed: local or S3)
# ---------------------------------------------------------------------------

def save_checkpoint(storage: Any, run_id: str, loop_idx: int, payload: dict) -> None:
    storage.save_json(f"{run_id}_loop{loop_idx}.json", payload)


def load_latest_checkpoint(storage: Any, run_id: str) -> Optional[dict]:
    prefix = f"{run_id}_loop"
    files = storage.list_files(prefix=prefix)
    if not files:
        return None
    return storage.load_json(files[-1])


def append_metrics(storage: Any, key: str, entry: dict) -> None:
    try:
        data = storage.load_json(key)
    except FileNotFoundError:
        data = []
    data.append(entry)
    storage.save_json(key, data)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def sum_tokens(responses: list[GenerationResponse]) -> tuple[int, int]:
    return (
        sum(r.prompt_tokens for r in responses),
        sum(r.completion_tokens for r in responses),
    )


# ---------------------------------------------------------------------------
# Eval aggregation
# ---------------------------------------------------------------------------

def aggregate_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n_evaluated": 0}
    keys = {k for r in results for k in r if k != "pred_accuracies"}
    agg: dict[str, Any] = {"n_evaluated": len(results)}
    for k in sorted(keys):
        vals = [r[k] for r in results if k in r]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            agg[k] = round(float(np.mean(vals)), 4)
    return agg


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_problems(problems: list[dict[str, Any]]) -> None:
    for i, p in enumerate(problems):
        if "orig_prompt" not in p:
            raise ValueError(f"Problem {i} missing required field 'orig_prompt'")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_run_config(config_ref: str, include_path: Optional[str] = None) -> RunConfig:
    """Resolve a config by registry name, .py file, .yaml file, or .json file."""
    if include_path:
        _discover_configs(include_path)
    if config_ref in configs:
        return configs.get(config_ref)()
    if config_ref.endswith(".py"):
        return _load_py_config(config_ref)
    with open(config_ref, "r", encoding="utf-8") as f:
        if config_ref.endswith((".yaml", ".yml")):
            import yaml
            payload = yaml.safe_load(f)
        else:
            payload = json.load(f)
    return RunConfig(**payload)


def _load_py_config(path: str) -> RunConfig:
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.config


def _discover_configs(directory: str) -> None:
    for f in sorted(Path(directory).glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"_cfg_{f.stem}", str(f))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
