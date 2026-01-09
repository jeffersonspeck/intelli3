from __future__ import annotations

from typing import Sequence
import math

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two vectors.

    - Works with lists/tuples.
    - Uses numpy if available, otherwise pure-Python.
    """
    if np is not None:
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        denom = (np.linalg.norm(av) * np.linalg.norm(bv))
        if denom == 0:
            return 0.0
        return float(np.dot(av, bv) / denom)

    # fallback
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    denom = math.sqrt(na) * math.sqrt(nb)
    return 0.0 if denom == 0.0 else (dot / denom)
