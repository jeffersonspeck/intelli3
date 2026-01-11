from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

from .loader import load_from_output_dir
from .profiles import MI_ORDER, get_profiles, list_profiles
from .ranking import rank_documents

ABBR_TO_INDEX = {
    "LING": 0,
    "LOG": 1,
    "ESP": 2,
    "MUS": 3,
    "CORP": 4,
    "INTER": 5,
    "INTRA": 6,
    "NAT": 7,
}

def _parse_focus(focus_csv: Optional[str]) -> Optional[List[int]]:
    if not focus_csv:
        return None
    raw = [p.strip().upper() for p in focus_csv.split(",") if p.strip()]
    idxs: List[int] = []
    for tok in raw:
        if tok in ABBR_TO_INDEX:
            idxs.append(ABBR_TO_INDEX[tok])
            continue
        # also accept full MI names, e.g. "Linguistic"
        for i, name in enumerate(MI_ORDER):
            if tok == name.upper().replace("-", "_") or tok == name.upper():
                idxs.append(i)
                break
        else:
            raise ValueError(f"FOCUS inválido: '{tok}'. Use: {', '.join(ABBR_TO_INDEX.keys())}")
    # remove duplicates while keeping order
    seen=set()
    uniq=[]
    for i in idxs:
        if i not in seen:
            uniq.append(i); seen.add(i)
    return uniq or None

def _write_ranking_csv(path: Path, rows: List[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "doc_id", "score"])
        for i, (doc_id, score) in enumerate(rows, 1):
            w.writerow([i, doc_id, f"{score:.6f}"])

def _write_ranking_json(path: Path, rows: List[tuple[str, float]], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "ranking": [{"rank": i+1, "doc_id": d, "score": s} for i, (d, s) in enumerate(rows)]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _cli() -> int:
    ap = argparse.ArgumentParser(description="Intelli3 Classifier CLI (rank documents by MI similarity)")
    ap.add_argument("--output", default="output", help="Diretório raiz com output/<id>/instances_fragments_profile.ttl")
    ap.add_argument("--filename", default="instances_fragments_profile.ttl", help="Nome do TTL por documento")
    ap.add_argument("--profile", default=None, help="Perfil (ex: P11-GLOB) ou 'list'.")
    ap.add_argument("--profiles", default=None, help="Lista de perfis (ex: P1-LING,P2-LOG) para rodar em lote.")
    ap.add_argument("--focus", default=None, help="Subconjunto de inteligências (ex: LOG,ESP,NAT). Opcional.")
    ap.add_argument("--topk", type=int, default=10, help="Top-K por perfil")
    ap.add_argument("--limit", type=int, default=0, help="Limita N documentos (0=sem limite)")
    ap.add_argument("--save", action="store_true", help="Salva CSV/JSON em --report-dir")
    ap.add_argument("--report-dir", default="reports", help="Diretório para salvar relatórios")
    args = ap.parse_args()

    if args.profile and args.profile.strip().lower() == "list":
        print("Perfis disponíveis:", ", ".join(list_profiles()))
        return 0

    docs = load_from_output_dir(args.output, filename=args.filename)
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]

    print("=== Intelli3 Classifier CLI ===")
    print(f"[INFO] output='{args.output}'  docs={len(docs)}  filename='{args.filename}'")
    print(f"[INFO] MI_ORDER: {', '.join([f'{i}:{name}' for i, name in enumerate(MI_ORDER)])}")

    if not docs:
        print("[WARN] Nenhum documento carregado. Verifique o diretório output/ e o filename.")
        return 1

    # sanity checks (aligned with PoC)
    bad = [d for d in docs if len(d.vector) != 8]
    if bad:
        print(f"[WARN] {len(bad)} docs com vetor inválido (dim != 8). Ex.: {bad[0].source_path}")
    zeros = [d for d in docs if sum(d.vector) <= 0]
    if zeros:
        print(f"[WARN] {len(zeros)} docs com vetor soma=0. Ex.: {zeros[0].source_path}")

    focus = _parse_focus(args.focus)

    prof_keys: List[str]
    if args.profiles:
        prof_keys = [p.strip() for p in args.profiles.split(",") if p.strip()]
    elif args.profile:
        prof_keys = [args.profile.strip()]
    else:
        prof_keys = ["P11-GLOB"]

    profiles = get_profiles(prof_keys)

    # docs as (doc_id, vector)
    doc_list = [(d.doc_id, d.vector) for d in docs]

    report_root = Path(args.report_dir)
    for prof in profiles:
        ranked = rank_documents(prof.vector, doc_list, top_k=args.topk, focus=focus)
        print(f"\n--- Ranking for {prof.key} ({prof.label}) ---")
        if focus:
            inv = {v:k for k,v in ABBR_TO_INDEX.items()}
            focus_lbl = ",".join(inv[i] for i in focus)
            print(f"[INFO] focus={focus_lbl}")
        rows = [(r.doc_id, r.score) for r in ranked]
        for i, (doc_id, score) in enumerate(rows, 1):
            print(f"{i:02d}. score={score:.4f}  doc={doc_id}")

        if args.save:
            meta = {
                "output": args.output,
                "filename": args.filename,
                "profile": prof.key,
                "profile_label": prof.label,
                "topk": args.topk,
                "focus": focus,
                "mi_order": MI_ORDER,
            }
            safe_key = prof.key.replace("/", "_").replace(":", "_")
            _write_ranking_csv(report_root / f"ranking_{safe_key}.csv", rows)
            _write_ranking_json(report_root / f"ranking_{safe_key}.json", rows, meta)

    if args.save:
        print(f"\n[OK] Relatórios salvos em: {report_root.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(_cli())
