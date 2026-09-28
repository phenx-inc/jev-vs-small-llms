"""Small protocol helpers kept independent from the GPU stack for unit tests."""

from __future__ import annotations

from typing import Any


def extract_input_ids(encoded: Any) -> list[int]:
    """Normalize Transformers list/tensor/BatchEncoding chat-template outputs."""
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)) or any(
        not isinstance(token, int) for token in encoded
    ):
        raise TypeError(f"chat template did not produce token IDs: {type(encoded)!r}")
    return list(encoded)
