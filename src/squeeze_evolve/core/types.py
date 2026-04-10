"""Multimodal prompt type for vision benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class MultimodalPrompt:
    """A prompt carrying text and optional base64-encoded images.

    Images are stored as data URLs (``data:image/...;base64,...``) at
    original resolution — no resizing is applied.
    """

    text: str
    images: list[str] = field(default_factory=list)

    @staticmethod
    def from_raw(raw: Union[str, dict, "MultimodalPrompt"]) -> MultimodalPrompt:
        """Normalize various input formats into a ``MultimodalPrompt``.

        Accepted inputs:
        * ``str`` — text-only prompt.
        * ``dict`` with ``"text"`` and optional ``"images"`` keys.
        * An existing ``MultimodalPrompt`` (returned as-is).
        """
        if isinstance(raw, MultimodalPrompt):
            return raw
        if isinstance(raw, str):
            return MultimodalPrompt(text=raw)
        if isinstance(raw, dict):
            return MultimodalPrompt(
                text=raw.get("text", ""),
                images=raw.get("images", []),
            )
        raise TypeError(f"Cannot convert {type(raw).__name__} to MultimodalPrompt")

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


# Union type used throughout the pipeline: either a plain string (text-only
# benchmarks) or a MultimodalPrompt (vision benchmarks).
Prompt = Union[str, MultimodalPrompt]
