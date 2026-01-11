
"""
PT-BR: Suite de testes do PoC (carregamento + ranqueamento básico).
EN: PoC test suite (loading + basic ranking).
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from .loader import load_from_output_dir
from .profiles import get_profiles, list_profiles
from .ranking import rank_documents

def _cli() -> int:
    ap = argparse.ArgumentParser(description="Intelli3 PoC tests (load + ranking).")
    ap.add_argument("--output", default="output", help="Diretório raiz que contém output/<id>/instances_fragments_profile.ttl")
    ap.add_argument("--filename", default="instances_fragments_profile.ttl", help="Nome do TTL por documento.")
    ap.add_argument("--limit", type=int, default=0, help="Limita N documentos (0=sem limite).")
    ap.add_argument("--profiles", default="all", help="Lista de perfis (ex: P1-LING,P2-LOG) ou 'all'.")
    ap.add_argument("--topk", type=int, default=10, help="Top-K por perfil.")
    ap.add_argument("--only-load", action="store_true", help="Apenas testa carregamento (sem ranqueamento).")
    ap.add_argument("--list-profiles", action="store_true", help="Lista perfis disponíveis e sai.")

    args = ap.parse_args()

    if args.list_profiles:
        print("Perfis disponíveis:", ", ".join(list_profiles()))
        return 0

    docs = load_from_output_dir(args.output, filename=args.filename)
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]

    print("=== Intelli3 PoC Tests ===")
    print(f"[INFO] Loaded {len(docs)} documents from '{args.output}'")

    if not docs:
        print("[WARN] Not enough documents for ranking tests")
        return 0

    # sanity checks
    bad = [d for d in docs if len(d.vector) != 8]
    if bad:
        print(f"[WARN] {len(bad)} docs com vetor inválido (dim != 8). Ex.: {bad[0].source_path}")
    zeros = [d for d in docs if sum(d.vector) <= 0]
    if zeros:
        print(f"[WARN] {len(zeros)} docs com vetor soma=0. Ex.: {zeros[0].source_path}")

    if args.only_load:
        return 0

    prof_keys = [p.strip() for p in args.profiles.split(",")] if args.profiles else ["all"]
    profiles = get_profiles(prof_keys)

    # prepara docs como (doc_id, vector)
    doc_list: List[Tuple[str, List[float]]] = [(d.doc_id, d.vector) for d in docs]

    for prof in profiles:
        ranked = rank_documents(prof.vector, doc_list, top_k=args.topk)
        print(f"\n--- Ranking for {prof.key} ({prof.label}) ---")
        for i, r in enumerate(ranked, 1):
            print(f"{i:02d}. score={r.score:.4f}  doc={r.doc_id}")

    return 0

if __name__ == "__main__":
    raise SystemExit(_cli())
