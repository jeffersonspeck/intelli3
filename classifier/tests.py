"""
PT-BR: Suite de testes do PoC (carregamento + métricas de estabilidade/eficácia).
EN: PoC test suite (loading + stability/effectiveness metrics).

Este módulo espelha o protocolo descrito na dissertação:
- carrega vetores MI a partir de um ou mais diretórios output/ (runs)
- calcula distribuição de similaridades por perfil de referência (média, dp, min, max, amplitude)
- calcula estabilidade vetorial entre runs (cos médio + L2 médio)
- calcula estabilidade de ranqueamento (Spearman rho) entre runs
- opcionalmente calcula F1 micro/macro vs ground-truth (doc_id,label)
- exporta relatórios CSV/JSON para facilitar escrita da dissertação
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .loader import load_from_output_dir
from .profiles import MI_ORDER, get_profiles, list_profiles
from .ranking import rank_documents, cosine

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
            idxs.append(ABBR_TO_INDEX[tok]); continue
        for i, name in enumerate(MI_ORDER):
            if tok == name.upper().replace("-", "_") or tok == name.upper():
                idxs.append(i); break
        else:
            raise ValueError(f"FOCUS inválido: '{tok}'. Use: {', '.join(ABBR_TO_INDEX.keys())}")
    seen=set(); uniq=[]
    for i in idxs:
        if i not in seen:
            uniq.append(i); seen.add(i)
    return uniq or None

def _project(v: Sequence[float], focus: Optional[Iterable[int]]) -> List[float]:
    if not focus:
        return [float(x) for x in v]
    idxs = list(focus)
    return [float(v[i]) for i in idxs]

def l2(a: Sequence[float], b: Sequence[float], *, focus: Optional[Iterable[int]] = None) -> float:
    av = _project(a, focus); bv = _project(b, focus)
    return sum((x-y)*(x-y) for x,y in zip(av,bv)) ** 0.5

def spearman_rho(rank_a: List[str], rank_b: List[str]) -> float:
    """Spearman's rho between two rankings (lists of doc_ids)."""
    pos_a = {doc_id: i+1 for i, doc_id in enumerate(rank_a)}
    pos_b = {doc_id: i+1 for i, doc_id in enumerate(rank_b)}
    common = [d for d in rank_a if d in pos_b]
    n = len(common)
    if n < 2:
        return 0.0
    ra = [pos_a[d] for d in common]
    rb = [pos_b[d] for d in common]
    mean_a = sum(ra)/n
    mean_b = sum(rb)/n
    num = sum((x-mean_a)*(y-mean_b) for x,y in zip(ra,rb))
    den_a = sum((x-mean_a)**2 for x in ra) ** 0.5
    den_b = sum((y-mean_b)**2 for y in rb) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)

def dominant_label(vec: Sequence[float]) -> str:
    idx = max(range(len(vec)), key=lambda i: float(vec[i]))
    inv = {v:k for k,v in ABBR_TO_INDEX.items()}
    return inv[idx]

@dataclass(frozen=True)
class SimilarityStats:
    profile: str
    n_docs: int
    mean: float
    stdev: float
    min: float
    max: float
    amplitude: float

def similarity_stats(profile_vec: Sequence[float], docs: List[Tuple[str, List[float]]], *, focus: Optional[Iterable[int]] = None) -> SimilarityStats:
    scores = [cosine(_project(profile_vec, focus), _project(vec, focus)) for _, vec in docs]
    n = len(scores)
    if n == 0:
        return SimilarityStats(profile="", n_docs=0, mean=0.0, stdev=0.0, min=0.0, max=0.0, amplitude=0.0)
    mean = sum(scores) / n
    var = sum((s-mean)**2 for s in scores) / n
    stdev = var ** 0.5
    mn = min(scores); mx = max(scores)
    return SimilarityStats(profile="", n_docs=n, mean=mean, stdev=stdev, min=mn, max=mx, amplitude=(mx-mn))

def f1_micro(pred: List[str], true: List[str]) -> float:
    """Micro F1 for multiclass single-label (equivale a acurácia)."""
    if not pred or not true or len(pred) != len(true):
        return 0.0
    correct = sum(1 for p,t in zip(pred,true) if p == t)
    return correct / len(true)

def f1_macro(pred: List[str], true: List[str], *, labels: Optional[List[str]] = None) -> float:
    if not pred or not true or len(pred) != len(true):
        return 0.0
    if labels is None:
        labels = sorted(set(true) | set(pred))
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for p,t in zip(pred,true) if p==lab and t==lab)
        fp = sum(1 for p,t in zip(pred,true) if p==lab and t!=lab)
        fn = sum(1 for p,t in zip(pred,true) if p!=lab and t==lab)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0); continue
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp+fp) if (tp+fp) else 0.0
        rec  = tp / (tp+fn) if (tp+fn) else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        f1s.append(f1)
    return sum(f1s)/len(f1s) if f1s else 0.0

def load_ground_truth_csv(path: str) -> Dict[str, str]:
    """CSV com cabeçalhos: doc_id,label (label em {LING,LOG,ESP,MUS,CORP,INTER,INTRA,NAT})."""
    m: Dict[str, str] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            doc_id = (row.get("doc_id") or "").strip()
            lab = (row.get("label") or "").strip().upper()
            if not doc_id or not lab:
                continue
            if lab not in ABBR_TO_INDEX:
                for abbr, idx in ABBR_TO_INDEX.items():
                    if lab in {MI_ORDER[idx].upper(), MI_ORDER[idx].upper().replace("-", "_")}:
                        lab = abbr
                        break
            if lab not in ABBR_TO_INDEX:
                raise ValueError(f"Rótulo desconhecido no ground-truth: '{lab}' (doc_id={doc_id})")
            m[doc_id] = lab
    return m

@dataclass
class RunData:
    name: str
    docs: List[Tuple[str, List[float]]]

def _load_run(output_dir: str, filename: str, limit: int = 0) -> RunData:
    docs_loaded = load_from_output_dir(output_dir, filename=filename)
    if limit and limit > 0:
        docs_loaded = docs_loaded[:limit]
    docs = [(d.doc_id, d.vector) for d in docs_loaded]
    return RunData(name=output_dir, docs=docs)

def _write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)

def _cli() -> int:
    ap = argparse.ArgumentParser(description="Intelli3 PoC tests (stability + ranking + optional F1)")
    ap.add_argument("--outputs", default="output", help="Um ou mais diretórios de output (separados por vírgula)")
    ap.add_argument("--filename", default="instances_fragments_profile.ttl", help="Nome do TTL por documento")
    ap.add_argument("--limit", type=int, default=0, help="Limita N docs por run (0=sem limite)")
    ap.add_argument("--profiles", default="all", help="Lista de perfis (ex: P1-LING,P2-LOG) ou 'all'")
    ap.add_argument("--topk", type=int, default=10, help="Top-K por perfil")
    ap.add_argument("--focus", default=None, help="Subconjunto de inteligências (ex: LOG,ESP,NAT). Opcional")
    ap.add_argument("--ground-truth", default=None, help="CSV doc_id,label para F1 (opcional)")
    ap.add_argument("--save", action="store_true", help="Gera arquivos em --report-dir")
    ap.add_argument("--report-dir", default="reports", help="Diretório para salvar relatórios")
    ap.add_argument("--list-profiles", action="store_true", help="Lista perfis disponíveis e sai")
    args = ap.parse_args()

    if args.list_profiles:
        print("Perfis disponíveis:", ", ".join(list_profiles()))
        return 0

    outputs = [p.strip() for p in args.outputs.split(",") if p.strip()]
    focus = _parse_focus(args.focus)
    runs = [_load_run(o, args.filename, limit=args.limit) for o in outputs]

    print("=== Intelli3 PoC Tests ===")
    print(f"[INFO] runs={len(runs)}  outputs={outputs}")
    print(f"[INFO] filename='{args.filename}'  limit={args.limit or 'all'}")
    print(f"[INFO] MI_ORDER: {', '.join([f'{i}:{name}' for i, name in enumerate(MI_ORDER)])}")
    if focus:
        inv = {v:k for k,v in ABBR_TO_INDEX.items()}
        print(f"[INFO] focus={','.join(inv[i] for i in focus)}")

    if not runs or not runs[0].docs:
        print("[WARN] Nenhum documento carregado. Verifique o diretório output/ e o filename.")
        return 1

    prof_keys = [p.strip() for p in args.profiles.split(",")] if args.profiles else ["all"]
    profiles = get_profiles(prof_keys)

    report_root = Path(args.report_dir)

    all_summary: Dict[str, object] = {
        "mi_order": MI_ORDER,
        "outputs": outputs,
        "filename": args.filename,
        "limit": args.limit,
        "focus": focus,
        "profiles": [p.key for p in profiles],
    }

    # 1) distribuição de similaridades + top-k por perfil/run
    sim_rows: List[List[object]] = [[ "run", "profile", "n_docs", "mean", "stdev", "min", "max", "amplitude" ]]
    for run in runs:
        doc_list = run.docs
        print(f"\n[RUN] {run.name}  docs={len(doc_list)}")
        for prof in profiles:
            stats = similarity_stats(prof.vector, doc_list, focus=focus)
            sim_rows.append([run.name, prof.key, stats.n_docs, f"{stats.mean:.6f}", f"{stats.stdev:.6f}", f"{stats.min:.6f}", f"{stats.max:.6f}", f"{stats.amplitude:.6f}"])
            ranked = rank_documents(prof.vector, doc_list, top_k=args.topk, focus=focus)
            print(f"  - {prof.key}: mean_cos={stats.mean:.3f}  stdev={stats.stdev:.3f}  top1={ranked[0].doc_id if ranked else 'N/A'}")
            if args.save:
                rows = [[i+1, r.doc_id, f"{r.score:.6f}"] for i, r in enumerate(ranked)]
                _write_csv(report_root / "topk" / f"topk_{run.name.replace('/', '_')}_{prof.key}.csv", ["rank","doc_id","score"], rows)

    if args.save:
        _write_csv(report_root / "similarity_stats.csv", sim_rows[0], sim_rows[1:])

    # 2) estabilidade (se houver ao menos 2 runs)
    if len(runs) >= 2:
        base_run = runs[0]
        other = runs[1]

        base_map = {doc_id: vec for doc_id, vec in base_run.docs}
        other_map = {doc_id: vec for doc_id, vec in other.docs}
        common = [doc_id for doc_id in base_map.keys() if doc_id in other_map]
        print(f"\n[STABILITY] Comparing runs: '{base_run.name}' vs '{other.name}'  common_docs={len(common)}")

        if common:
            cos_vals = [cosine(_project(base_map[d], focus), _project(other_map[d], focus)) for d in common]
            l2_vals  = [l2(base_map[d], other_map[d], focus=focus) for d in common]
            mean_cos = sum(cos_vals)/len(cos_vals)
            mean_l2  = sum(l2_vals)/len(l2_vals)
            print(f"  - Vector stability: mean_cos={mean_cos:.6f}  mean_l2={mean_l2:.6f}")
            all_summary["vector_stability"] = {
                "runs": [base_run.name, other.name],
                "common_docs": len(common),
                "mean_cosine": mean_cos,
                "mean_l2": mean_l2,
            }
            if args.save:
                rows = [[d, f"{cosine(_project(base_map[d], focus), _project(other_map[d], focus)):.6f}", f"{l2(base_map[d], other_map[d], focus=focus):.6f}"] for d in common]
                _write_csv(report_root / "stability_vectors.csv", ["doc_id","cosine","l2"], rows)

        rho_rows = [[ "profile","rho_spearman","topk" ]]
        for prof in profiles:
            r1 = rank_documents(prof.vector, base_run.docs, top_k=args.topk, focus=focus)
            r2 = rank_documents(prof.vector, other.docs, top_k=args.topk, focus=focus)
            rho = spearman_rho([r.doc_id for r in r1], [r.doc_id for r in r2])
            rho_rows.append([prof.key, f"{rho:.6f}", args.topk])
            print(f"  - Ranking stability {prof.key}: rho={rho:.4f} (topk={args.topk})")
        if args.save:
            _write_csv(report_root / "stability_rankings.csv", rho_rows[0], rho_rows[1:])
            all_summary["ranking_stability"] = [{"profile": row[0], "rho": float(row[1]), "topk": args.topk} for row in rho_rows[1:]]

    # 3) eficácia vs ground truth (opcional)
    if args.ground_truth:
        gt = load_ground_truth_csv(args.ground_truth)
        preds: List[str] = []
        trues: List[str] = []
        for doc_id, vec in runs[0].docs:
            if doc_id not in gt:
                continue
            preds.append(dominant_label(vec))
            trues.append(gt[doc_id])
        micro = f1_micro(preds, trues)
        macro = f1_macro(preds, trues, labels=list(ABBR_TO_INDEX.keys()))
        print(f"\n[EFFECTIVENESS] ground-truth='{args.ground_truth}'  n={len(trues)}")
        print(f"  - F1-micro={micro:.4f}  F1-macro={macro:.4f}")
        all_summary["effectiveness"] = {
            "ground_truth": args.ground_truth,
            "n": len(trues),
            "f1_micro": micro,
            "f1_macro": macro,
        }
        if args.save:
            _write_csv(report_root / "effectiveness_predictions.csv", ["doc_id","true","pred"], [[doc_id, gt[doc_id], dominant_label(vec)] for doc_id, vec in runs[0].docs if doc_id in gt])

    if args.save:
        (report_root / "poc_summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] Relatórios salvos em: {report_root.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(_cli())
