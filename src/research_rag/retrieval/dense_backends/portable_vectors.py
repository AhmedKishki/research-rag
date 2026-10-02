"""The one validation a portable vector file passes before either backend reads it.

Both backends accept the same generation-relative `float32` matrix, and each
carried its own copy of the checks that refuse everything else, so a file one
backend accepted and the other refused was a difference in the copies rather than
in the file. The two shape contracts are not the same check and keep their own
messages: `count` is the row count a chunk collection demands, and without it only
the width is known and the row count is settled by the caller that owns it.

This module may never import a backend, a model, or a collection, because a check
that decides what is readable must not decide what is searched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Values are checked in strides rather than whole, because the file is mapped and a
# corpus can be larger than the memory reading it at once.
_FINITE_CHECK_STRIDE = 4096


def load_portable_vectors(
    vectors_path: Path,
    dimension: int,
    *,
    count: int | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """The mapped float32 matrix at `vectors_path`, or the reason it is refused.

    With `count` the file must hold exactly that many rows of `dimension` values;
    without it, every row must be `dimension` wide and the caller settles how many
    rows it expects.
    """

    try:
        vectors = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load portable embeddings: {vectors_path}") from exc
    if vectors.dtype != np.float32:
        raise ValueError("Portable embeddings must use float32 values")
    if count is None:
        invalid = vectors.ndim != 2 or vectors.shape[1] != dimension
    else:
        invalid = vectors.ndim != 2 or vectors.shape != (count, dimension)
    if invalid:
        raise ValueError(
            "Portable embedding dimensions do not match the chunk collection"
            if count is not None
            else "Portable embeddings have an invalid shape"
        )
    for offset in range(0, vectors.shape[0], _FINITE_CHECK_STRIDE):
        if not np.isfinite(vectors[offset : offset + _FINITE_CHECK_STRIDE]).all():
            raise ValueError("Portable embeddings contain non-finite values")
    return vectors
