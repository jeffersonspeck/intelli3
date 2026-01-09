from __future__ import annotations

import argparse
from pathlib import Path

from .loader import load_vectors
from .ranking import rank


def main() -> int:
    parser = argparse.ArgumentParser(description="Intelli3 simple CLI (rank documents from output/)")
    parser.add_argument("--output", default="output", help="Path to output/ directory")
    parser.add_argument("--query", default=None, help="Doc id (URI) to use as query. Default: first loaded doc.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    docs = load_vectors(Path(args.output))
    if not docs:
        print("No documents loaded. Check output/<id>/instances_fragments_profile.ttl")
        return 1

    cand = [{"doc_id": d["doc_id"], "vector": d["vector"]} for d in docs]
    q = next((d for d in docs if d["doc_id"] == args.query), docs[0])

    print(f"Query: {q['doc_id']}  primary={q['primary']}")
    for i, (doc_id, sim) in enumerate(rank(q["vector"], cand, top_k=args.top_k), start=1):
        print(f"{i:02d}. {doc_id}  sim={sim:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
