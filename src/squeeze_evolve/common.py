"""Shared utilities and reusable operator implementations.

Nothing is registered here. All operator registration happens in
benchmarks/*/register.py using @registry.register() decorators.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Text utility
# ---------------------------------------------------------------------------

def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> chain-of-thought wrappers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ---------------------------------------------------------------------------
# Math answer extraction
# ---------------------------------------------------------------------------

def _extract_boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    idx = start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        idx += 1
    return None


def _extract_tagged_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_final_answer_line(text: str) -> str | None:
    match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def _strip_math_wrappers(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.replace("\\(", "").replace("\\)", "")
    stripped = stripped.replace("\\[", "").replace("\\]", "")
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        stripped = stripped[1:-1]
    return stripped.strip()


def _normalize_math_answer(text: str) -> str:
    stripped = _strip_math_wrappers(text)
    stripped = stripped.replace(",", "")
    stripped = re.sub(r"\s+", "", stripped)
    stripped = stripped.strip(".")
    if stripped.startswith("\\text{") and stripped.endswith("}"):
        stripped = stripped[6:-1].strip()
    numeric_match = re.fullmatch(r"[-+]?\d+", stripped)
    if numeric_match:
        return str(int(stripped))
    return stripped.lower()


def extract_boxed_math_answer(candidate: str) -> str:
    text = strip_think_blocks(candidate or "")
    boxed = _extract_boxed_content(text)
    if boxed is not None:
        return _normalize_math_answer(boxed)
    tagged = _extract_tagged_answer(text)
    if tagged is not None:
        return _normalize_math_answer(tagged)
    final_line = _extract_final_answer_line(text)
    if final_line is None:
        return ""
    inline_boxed = _extract_boxed_content(final_line)
    if inline_boxed is not None:
        return _normalize_math_answer(inline_boxed)
    number_match = re.search(r"[-+]?\d+", final_line.replace(",", ""))
    if number_match:
        return str(int(number_match.group(0)))
    return _normalize_math_answer(final_line)


def eval_boxed_math(candidates: list[str], gt: Any, **kwargs: Any) -> dict[str, Any]:
    """Math scoring using boxed/final-answer extraction and normalized voting."""
    gt_str = _normalize_math_answer(str(gt))
    extracted = [extract_boxed_math_answer(candidate) for candidate in candidates]
    correct = [answer == gt_str for answer in extracted]
    counts = Counter(extracted)
    majority = counts.most_common(1)[0][0] if counts else ""
    return {
        "pred_accuracies": [float(c) for c in correct],
        "mean_acc": float(sum(correct)) / max(1, len(correct)),
        "pass_at_k": float(any(correct)),
        "majority_vote": float(majority == gt_str),
    }


# ---------------------------------------------------------------------------
# Recombination factories
# ---------------------------------------------------------------------------

def make_aggregate_prompt(
    kind: str,
    answer_format: str,
    is_code: bool = False,
    custom_fn: Callable[[str, list[str]], str] | None = None,
) -> Callable[..., str]:
    """Create a task-aware aggregate recombination function."""
    def aggregate(query: str, candidates: list[str], **kwargs: Any) -> str:
        if not candidates:
            return query
        if custom_fn is not None:
            return custom_fn(query, candidates)

        parts: list[str] = []
        if len(candidates) == 1:
            if is_code:
                parts.append(
                    f"You are given a {kind} and a candidate solution. "
                    "The candidate may be incomplete or contain errors. "
                    "Refine this code and produce an improved solution. "
                    "If it is entirely wrong, attempt a new strategy. "
                    f"Return your final code in {answer_format}.\n"
                )
            else:
                parts.append(
                    f"You are given a {kind} and a candidate solution. "
                    "The candidate may be incomplete or contain errors. "
                    "Refine this trajectory and produce an improved, higher-quality solution. "
                    "If it is entirely wrong, attempt a new strategy. "
                    f"End with the final result in {answer_format}.\n"
                )
        else:
            if is_code:
                parts.append(
                    f"You are given a {kind} and several candidate solutions. "
                    "Some candidates may be better than others. "
                    "Combine the best ideas and fix issues in weaker solutions. "
                    "Produce a single, improved solution. "
                    f"Return your final code in {answer_format}.\n"
                )
            else:
                parts.append(
                    f"You are given a {kind} and several candidate solutions. "
                    "Some candidates may be incorrect or contain errors. "
                    "Aggregate the useful ideas and produce a single, high-quality solution. "
                    "Reason carefully; if candidates disagree, choose the correct path. "
                    "If all are incorrect, then attempt a different strategy. "
                    f"End with the final result in {answer_format}.\n"
                )

        parts.append("Problem:\n")
        parts.append(query.strip() + "\n")

        if len(candidates) == 1:
            parts.append("Candidate solution (may contain mistakes):\n")
            parts.append(f"---- Candidate ----\n{(candidates[0] or '').strip()}\n")
            verb = "return the final code" if is_code else "end with the final answer"
            parts.append(f"Now refine the candidate into an improved solution. Provide clear reasoning and {verb} in {answer_format}.")
        else:
            parts.append("Candidate solutions (may contain mistakes):\n")
            for i, ans in enumerate(candidates, 1):
                parts.append(f"---- Solution {i} ----\n{(ans or '').strip()}\n")
            verb = "return the final code" if is_code else "end with the final answer"
            parts.append(f"Now write a single improved solution. Provide clear reasoning and {verb} in {answer_format}.")

        return "\n".join(parts)

    return aggregate


def synthesize_prompt(query: str, candidates: list[str], **kwargs: Any) -> str:
    """Minimal recombination prompt (no task awareness)."""
    if not candidates:
        return query
    joined = "\n\n".join(f"Candidate {i + 1}: {c}" for i, c in enumerate(candidates))
    return f"{query}\n\n{joined}\n\nSynthesize the best final answer."


# ---------------------------------------------------------------------------
# Evaluation implementations
# ---------------------------------------------------------------------------

def eval_exact_match(candidates: list[str], gt: Any, **kwargs: Any) -> dict[str, Any]:
    """Exact string equality against ground truth."""
    gt_str = str(gt).strip()
    correct = [c.strip() == gt_str for c in candidates]
    majority = Counter(c.strip() for c in candidates).most_common(1)[0][0] if candidates else ""
    return {
        "pred_accuracies": [float(c) for c in correct],
        "mean_acc": float(sum(correct)) / max(1, len(correct)),
        "pass_at_k": float(any(correct)),
        "majority_vote": float(majority == gt_str),
    }


def eval_none(candidates: list[str], gt: Any, **kwargs: Any) -> dict[str, Any]:
    """No evaluation (open-ended tasks without ground truth)."""
    return {
        "pred_accuracies": [],
        "mean_acc": 0.0,
        "pass_at_k": 0.0,
        "majority_vote": 0.0,
    }
