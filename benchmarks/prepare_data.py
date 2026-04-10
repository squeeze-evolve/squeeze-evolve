#!/usr/bin/env python3
"""Download and prepare multimodal benchmark datasets.

Converts HuggingFace datasets into self-contained parquet files with
base64-encoded images (original resolution, no resize).

Question formatting is aligned with lmms-eval-new task configs:
- BabyVision: post_prompt appended, choice questions formatted with (A)/(B)/...
- MMMU Pro: options formatted as A./B./..., post_prompt appended

Usage:
    python benchmarks/prepare_data.py --benchmark babyvision
    python benchmarks/prepare_data.py --benchmark mmmu_pro
    python benchmarks/prepare_data.py --benchmark all
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import re
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


# ---------------------------------------------------------------------------
# BabyVision
# Aligned with lmms-eval-new/lmms_eval/tasks/babyvision/
# ---------------------------------------------------------------------------

# Post prompt from babyvision.yaml
BABYVISION_POST_PROMPT = "\nThink about the question and give your final answer in \\boxed{Answer} format."


def _format_babyvision_question(doc: dict) -> str:
    """Format BabyVision question with choices (if applicable) and post-prompt.

    Aligned with babyvision/utils.py::babyvision_doc_to_text.
    """
    question = doc.get("question", "")
    ans_type = doc.get("ansType", "")

    # Add choices for multiple-choice questions
    if ans_type == "choice" and doc.get("options"):
        options = doc["options"]
        options_str = "\n".join(
            f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
        )
        return f"{question}\nChoices:\n{options_str}{BABYVISION_POST_PROMPT}"

    return f"{question}{BABYVISION_POST_PROMPT}"


def _format_babyvision_answer(doc: dict) -> str:
    """Extract ground truth answer from BabyVision doc.

    Aligned with babyvision/utils.py::babyvision_doc_to_target.
    """
    ans_type = doc.get("ansType", "")
    if ans_type == "choice":
        choice_ans = doc.get("choiceAns")
        if choice_ans is not None:
            return chr(ord("A") + int(choice_ans))
    else:
        blank_ans = doc.get("blankAns")
        if blank_ans is not None:
            return str(blank_ans)
    return ""


def prepare_babyvision(output_dir: str = "data/babyvision") -> None:
    """Download and prepare BabyVision dataset.

    Source: UnipatAI/BabyVision on HuggingFace.
    """
    from datasets import load_dataset

    print("Downloading BabyVision dataset...")
    ds = load_dataset("UnipatAI/BabyVision", split="train")

    rows = []
    for i, example in enumerate(ds):
        image = example.get("image", None)
        if image is None:
            print(f"  Skipping example {i}: no image")
            continue

        image_url = pil_to_data_url(image.convert("RGB"))
        question_text = _format_babyvision_question(example)
        answer = _format_babyvision_answer(example)

        rows.append({
            "prompt": question_text,
            "image": image_url,
            "reward_model": {"ground_truth": answer},
            # Store raw question for judge prompt (without post_prompt)
            "raw_question": example.get("question", ""),
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"BabyVision: {len(rows)} examples -> {out_path}")


# ---------------------------------------------------------------------------
# MMMU Pro
# Aligned with lmms-eval-new/lmms_eval/tasks/mmmu_pro/
# ---------------------------------------------------------------------------

# Post prompt from mmmu_pro_standard.yaml (default)
MMMU_PRO_POST_PROMPT = "Answer with the option letter from the given choices directly."


def _replace_image_tokens(text: str) -> str:
    """Replace <image 1>, <image 2>, ... with <image>."""
    for i in range(1, 8):
        text = text.replace(f"<image {i}>", "<image>")
    return text


def _format_mmmu_pro_question(doc: dict) -> str:
    """Format MMMU Pro question with options and post-prompt.

    Aligned with mmmu_pro/utils.py::construct_prompt + mmmu_pro_doc_to_text.
    """
    question = doc.get("question", "")
    options_raw = doc.get("options", "[]")

    if isinstance(options_raw, str):
        try:
            options = ast.literal_eval(options_raw)
        except (ValueError, SyntaxError):
            options = []
    else:
        options = list(options_raw)

    options_str = "\n".join(
        f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
    )
    prompt = f"{question}\n{options_str}\n\n{MMMU_PRO_POST_PROMPT}"
    return prompt


def prepare_mmmu_pro(output_dir: str = "data/mmmu_pro") -> None:
    """Download and prepare MMMU Pro dataset.

    Source: MMMU/MMMU_Pro on HuggingFace, "standard (10 options)" config.
    """
    from datasets import load_dataset

    print("Downloading MMMU Pro dataset...")
    try:
        ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    except Exception:
        try:
            ds = load_dataset("MMMU/MMMU_Pro", split="test")
        except Exception:
            ds = load_dataset("MMMU/MMMU_Pro", "vision", split="test")

    rows = []
    for i, example in enumerate(ds):
        question_text = _format_mmmu_pro_question(example)
        answer = example.get("answer", "")
        options_raw = example.get("options", "[]")

        # Collect images: check image_1 .. image_7 (referenced in question text)
        images = {}
        prompt_for_images = example.get("question", "") + str(options_raw)
        for img_idx in range(1, 8):
            img_key = f"image_{img_idx}"
            if f"<image {img_idx}>" in prompt_for_images:
                img = example.get(img_key, None)
                if img is not None:
                    images[img_key] = pil_to_data_url(img.convert("RGB"))

        # Also check for a single "image" column (vision-only variant)
        if not images and example.get("image") is not None:
            images["image"] = pil_to_data_url(example["image"].convert("RGB"))

        row = {
            "prompt": question_text,
            "reward_model": {"ground_truth": str(answer)},
            "options": options_raw if isinstance(options_raw, str) else json.dumps(options_raw),
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
