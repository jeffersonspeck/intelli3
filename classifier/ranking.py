
"""
PT-BR: Similaridade e ranqueamento no espaço vetorial MI.
EN: Similarity and ranking in the MI vector space.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import math

def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) * float(x) for x in v)))

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = l2_norm(a)
    nb = l2_norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return sum((float(x)*float(y) for x,y in zip(a,b))) / (na * nb)

@dataclass(frozen=True)
class RankedDoc:
    doc_id: str
    score: float

def rank_documents(profile_vec: Sequence[float], docs: List[Tuple[str, List[float]]], *, top_k: int = 10) -> List[RankedDoc]:
    ranked = [RankedDoc(doc_id=d_id, score=cosine(profile_vec, vec)) for d_id, vec in docs]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:max(1, int(top_k))]
