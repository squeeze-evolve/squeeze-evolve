#!/usr/bin/env python3
"""Download and prepare multimodal benchmark datasets.

Converts HuggingFace datasets into self-contained parquet files with
base64-encoded images (original resolution, no resize).

Usage:
    python benchmarks/prepare_data.py --benchmark babyvision
    python benchmarks/prepare_data.py --benchmark mmmu_pro
    python benchmarks/prepare_data.py --benchmark all
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd


def pil_to_data_url(pil_image) -> str:
    """Convert a PIL Image to a base64 data URL (original resolution)."""
    buf = io.BytesIO()
    fmt = getattr(pil_image, "format", None) or "PNG"
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def prepare_babyvision(output_dir: str = "data/babyvision") -> None:
    """Download and prepare BabyVision dataset.

    Source: UnipatAI/BabyVision on HuggingFace.
    Schema: prompt (str), image (base64 data URL), reward_model.ground_truth (str)
    """
    from datasets import load_dataset

    print("Downloading BabyVision dataset...")
    ds = load_dataset("UnipatAI/BabyVision", split="test")

    rows = []
    for i, example in enumerate(ds):
        # Extract fields — adjust column names based on actual dataset schema
        question = example.get("question", example.get("prompt", ""))
        answer = example.get("answer", example.get("ground_truth", ""))
        image = example.get("image", None)

        if image is None:
            print(f"  Skipping example {i}: no image")
            continue

        image_url = pil_to_data_url(image)

        rows.append({
            "prompt": question,
            "image": image_url,
            "reward_model": {"ground_truth": str(answer)},
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test.parquet")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"BabyVision: {len(rows)} examples -> {out_path}")


def prepare_mmmu_pro(output_dir: str = "data/mmmu_pro") -> None:
    """Download and prepare MMMU Pro dataset.

    Source: MMMU/MMMU_Pro on HuggingFace.
    Schema: prompt (str), image_1..image_7 (base64 data URLs),
            reward_model.ground_truth (str), options (JSON list)
    """
    from datasets import load_dataset

    print("Downloading MMMU Pro dataset...")
    # MMMU Pro has multiple subsets; use the "standard" vision subset
    try:
        ds = load_dataset("MMMU/MMMU_Pro", "standard (no vision)", split="test")
    except Exception:
        # Try alternative config names
        try:
            ds = load_dataset("MMMU/MMMU_Pro", split="test")
        except Exception:
            ds = load_dataset("MMMU/MMMU_Pro", "vision", split="test")

    rows = []
    for i, example in enumerate(ds):
        question = example.get("question", example.get("prompt", ""))
        answer = example.get("answer", example.get("ground_truth", ""))
        options = example.get("options", [])

        # Collect images (up to 7)
        images = {}
        for img_idx in range(1, 8):
            img_key = f"image_{img_idx}"
            img = example.get(img_key, None)
            if img is not None:
                images[img_key] = pil_to_data_url(img)

        # Also check for a single "image" column
        if not images and example.get("image") is not None:
            images["image"] = pil_to_data_url(example["image"])

        # Serialize options as JSON string for parquet storage
        if isinstance(options, (list, tuple)):
            options_json = json.dumps(list(options))
        else:
            options_json = str(options)

        row = {
            "prompt": question,
            "reward_model": {"ground_truth": str(answer)},
            "options": options_json,
        }
        row.update(images)
        rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test.parquet")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"MMMU Pro: {len(rows)} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare multimodal benchmark datasets")
    parser.add_argument(
        "--benchmark",
        choices=["babyvision", "mmmu_pro", "all"],
        required=True,
        help="Which benchmark to prepare",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: data/<benchmark>/)",
    )
    args = parser.parse_args()

    if args.benchmark in ("babyvision", "all"):
        out = args.output_dir or "data/babyvision"
        prepare_babyvision(out)

    if args.benchmark in ("mmmu_pro", "all"):
        out = args.output_dir or "data/mmmu_pro"
        prepare_mmmu_pro(out)


if __name__ == "__main__":
    main()
