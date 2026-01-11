"""PT-BR: Similaridade e ranqueamento no espaço vetorial MI.
EN: Similarity and ranking in the MI vector space.

Design notes:
- This module is intentionally lightweight and has no dependency on the ontology.
- The canonical MI vector order is defined in `profiles.MI_ORDER`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional
import math

def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) * float(x) for x in v)))

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity (pure-Python, no deps)."""
    na = l2_norm(a)
    nb = l2_norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return sum((float(x) * float(y) for x, y in zip(a, b))) / (na * nb)

def project(v: Sequence[float], focus: Optional[Iterable[int]]) -> List[float]:
    """Project a vector onto selected dimensions (by index)."""
    if not focus:
        return [float(x) for x in v]
    idxs = list(focus)
    return [float(v[i]) for i in idxs]

@dataclass(frozen=True)
class RankedDoc:
    doc_id: str
    score: float

def rank_documents(
    profile_vec: Sequence[float],
    docs: List[Tuple[str, List[float]]],
    *,
    top_k: int = 10,
    focus: Optional[Iterable[int]] = None,
) -> List[RankedDoc]:
    """
    Rank documents by cosine similarity to a profile vector.

    Args:
        profile_vec: MI vector for the profile/query.
        docs: list of tuples (doc_id, vector)
        top_k: number of top results to return.
        focus: optional iterable of indices to restrict comparison (projection).
    """
    pv = project(profile_vec, focus)
    ranked = [RankedDoc(doc_id=d_id, score=cosine(pv, project(vec, focus))) for d_id, vec in docs]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[: max(1, int(top_k))]
