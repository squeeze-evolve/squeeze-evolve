"""Dataset loading and normalization.

Converts parquet / JSONL datasets into the ``[{orig_prompt, gt}]`` format
expected by :class:`RoutingOrchestrator`.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .types import MultimodalPrompt, Prompt


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


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------

def _pil_to_data_url(pil_image: Any) -> str:
    """Convert a PIL Image to a base64 data URL (original resolution, no resize)."""
    buf = io.BytesIO()
    fmt = getattr(pil_image, "format", None) or "PNG"
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _bytes_to_data_url(raw_bytes: bytes, mime: str = "image/png") -> str:
    """Convert raw image bytes to a base64 data URL."""
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_multimodal_prompt(row: dict) -> MultimodalPrompt:
    """Build a MultimodalPrompt from a parquet row.

    Collects images from columns named ``image``, ``image_1``, ``image_2``, etc.
    Each image value can be:
    * A base64 data URL string (``data:image/...;base64,...``).
    * A PIL Image object.
    * Raw ``bytes``.
    """
    text = _extract_prompt(row.get("prompt", ""))
    images: list[str] = []

    # Collect image columns: "image", then "image_1" .. "image_7"
    img_cols = ["image"] + [f"image_{i}" for i in range(1, 8)]
    for col in img_cols:
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, str) and val.startswith("data:"):
            images.append(val)
        elif isinstance(val, bytes):
            images.append(_bytes_to_data_url(val))
        else:
            # Assume PIL Image
            try:
                images.append(_pil_to_data_url(val))
            except Exception:
                pass  # skip unrecognized image types

    return MultimodalPrompt(text=text, images=images)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_parquet(
    path: str,
    n_problems: Optional[int] = None,
    multimodal: bool = False,
) -> list[dict[str, Any]]:
    """Load a parquet dataset (aime25, hmmt25, gpqa_diamond, babyvision, mmmu_pro, ...).

    Returns list of ``{orig_prompt, gt}`` dicts.  When *multimodal* is
    ``True``, ``orig_prompt`` is a :class:`MultimodalPrompt` instead of
    a plain string.
    """
    df = pd.read_parquet(path)
    if n_problems is not None:
        df = df.head(n_problems)
    problems = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if multimodal:
            prompt: Prompt = _extract_multimodal_prompt(row_dict)
        else:
            prompt = _extract_prompt(row_dict.get("prompt", ""))

        entry: dict[str, Any] = {
            "orig_prompt": prompt,
            "gt": _extract_gt(row_dict),
        }
        # Carry forward extra metadata for judge prompts
        if "options" in row_dict:
            entry["options"] = row_dict["options"]
        if "raw_question" in row_dict:
            entry["question"] = row_dict["raw_question"]
        problems.append(entry)
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


def load_dataset(
    path: str,
    n_problems: Optional[int] = None,
    multimodal: bool = False,
) -> list[dict[str, Any]]:
    """Auto-detect format and load a dataset.

    Supports ``.parquet``, ``.jsonl``, and ``.json`` files.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return load_parquet(path, n_problems, multimodal=multimodal)
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
