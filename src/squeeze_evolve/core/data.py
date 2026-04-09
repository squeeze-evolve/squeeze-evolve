"""Dataset loading and normalization.

Converts parquet / JSONL datasets into the ``[{orig_prompt, gt}]`` format
expected by :class:`RoutingOrchestrator`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _extract_prompt(cell: Any) -> str:
    """Extract user prompt from chat-message list or raw string."""
    if isinstance(cell, (list, tuple)):
        for msg in cell:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
        if cell and isinstance(cell[0], dict):
            return str(cell[0].get("content", ""))
        return str(cell[0]) if cell else ""
    return str(cell)


def _extract_gt(row: dict) -> Optional[str]:
    """Extract ground-truth answer from reward_model dict."""
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return None


def load_parquet(path: str, n_problems: Optional[int] = None) -> list[dict[str, Any]]:
    """Load a parquet dataset (aime25, hmmt25, gpqa_diamond, rg_*, supergpqa).

    Returns list of ``{orig_prompt, gt}`` dicts.
    """
    df = pd.read_parquet(path)
    if n_problems is not None:
        df = df.head(n_problems)
    problems = []
    for _, row in df.iterrows():
        problems.append({
            "orig_prompt": _extract_prompt(row["prompt"]),
            "gt": _extract_gt(row.to_dict()),
        })
    return problems


def load_jsonl(path: str, n_problems: Optional[int] = None) -> list[dict[str, Any]]:
    """Load a JSONL file as a list of dicts.

    Each line is parsed as JSON and returned as-is.
    """
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if n_problems is not None and len(problems) >= n_problems:
                break
            problems.append(json.loads(line))
    return problems


def load_dataset(path: str, n_problems: Optional[int] = None) -> list[dict[str, Any]]:
    """Auto-detect format and load a dataset.

    Supports ``.parquet``, ``.jsonl``, and ``.json`` files.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return load_parquet(path, n_problems)
    if p.suffix == ".jsonl":
        return load_jsonl(path, n_problems)
    if p.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if n_problems is not None:
            data = data[:n_problems]
        return data
    raise ValueError(f"Unsupported file format: {p.suffix}")


def list_datasets(data_dir: str = "data") -> list[str]:
    """List available datasets by scanning subdirectories for parquet/jsonl files."""
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(
        str(f.relative_to(root))
        for f in root.rglob("*")
        if f.suffix in (".parquet", ".jsonl", ".json") and f.is_file()
    )
