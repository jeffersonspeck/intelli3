from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from decimal import Decimal, ROUND_HALF_UP

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal, XSD
from rdflib.namespace import OWL

import json
from pathlib import Path
from typing import Any

# Namespaces exatamente como na OntoMI
BFO  = Namespace("http://purl.obolibrary.org/obo/BFO_")
IAO  = Namespace("http://purl.obolibrary.org/obo/IAO_")
ONTO = Namespace("https://techcoop.com.br/ontomi#")


# ------------------------------------------------------------
# Util
# ------------------------------------------------------------
def _base_from_fragment_uri(frag_uri: URIRef) -> str:
    u = str(frag_uri)
    return (u.split("#", 1)[0] + "#") if "#" in u else (u.rstrip("/") + "#")


def _qdec(x: float | int | str, places: int = 4) -> Decimal:
    d = Decimal(str(x))
    return d.quantize(Decimal("1." + "0"*places), rounding=ROUND_HALF_UP)


def _index_intelligences(g: Graph) -> Dict[str, URIRef]:
    """
    Indexa classes subclasses de onto:IntelligenceDisposition por:
      - localname (lower)
      - rdfs:label (lower)
    """
    idx: Dict[str, URIRef] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        if (cls, RDFS.subClassOf, ONTO.IntelligenceDisposition) in g:
            local = str(cls).rsplit("#", 1)[-1].lower()
            idx[local] = cls
            for _, _, lbl in g.triples((cls, RDFS.label, None)):
                idx[str(lbl).strip().lower()] = cls
    return idx


# ------------------------------------------------------------
# S4 — computa scores_by_fragment (a_{k,j}) diretamente do grafo
# ------------------------------------------------------------
def compute_scores_by_fragment_from_graph(
    g: Graph,
    fragment_nodes: List[URIRef],
    *,
    w_keyword: float = 1.00,
    w_context_object: float = 1.25,
    w_discursive_strategy: float = 1.10,
) -> Dict[int, Dict[str, float]]:
    """
    Constrói scores_by_fragment[k][intel_localname] = a_{k,j} a partir do grafo.

    Premissas (S3):
      - frag rdf:type onto:ExplanationFragment
      - frag onto:usesElement el
      - el rdf:type em {onto:Keyword, onto:ContextObject, onto:DiscursiveStrategy}
      - el onto:evokesIntelligence IntelligenceDispositionClass

    Retorna:
      dict[int, dict[str,float]] com chave de inteligência = localname da classe (ex.: "Linguistic").
    """
    role_weights = {
        ONTO.Keyword: float(w_keyword),
        ONTO.ContextObject: float(w_context_object),
        ONTO.DiscursiveStrategy: float(w_discursive_strategy),
    }

    scores_by_fragment: Dict[int, Dict[str, float]] = {}

    for k, frag in enumerate(fragment_nodes):
        per_intel: Dict[str, float] = {}

        for el in g.objects(frag, ONTO.usesElement):
            # Descobre role(e) pelo rdf:type e aplica w_role
            w_role: Optional[float] = None
            for t in g.objects(el, RDF.type):
                if t in role_weights:
                    w_role = role_weights[t]
                    break
            if w_role is None:
                continue

            # A evidência pode evocar 1+ inteligências
            for cls in g.objects(el, ONTO.evokesIntelligence):
                intel_name = str(cls).rsplit("#", 1)[-1]  # localname
                per_intel[intel_name] = per_intel.get(intel_name, 0.0) + float(w_role)

        if per_intel:
            scores_by_fragment[k] = per_intel

    return scores_by_fragment


def add_activations_rdf(
    g: Graph,
    fragment_nodes: List[URIRef],
    scores_by_fragment: Dict[int, Dict[str, float]],
    *,
    theta: float = 0.75,
    link_isAboutFragment: bool = True,
    vec_places: int = 4,
    clean_existing: bool = True,
) -> Tuple[Graph, Dict[int, List[URIRef]]]:
    """
    Garante 1 primária por fragmento e cardinalidade limpa para hasActivationType/hasPrimaryActivation.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta deve estar em (0,1].")

    intel_index = _index_intelligences(g)
    created_map: Dict[int, List[URIRef]] = {}

    for k, frag in enumerate(fragment_nodes):
        scores = scores_by_fragment.get(k, {})
        if not scores:
            continue

        # --- (Opcional, mas recomendado) limpa tudo que já estava ligado ao fragmento ---
        if clean_existing:
            old_acts = list(g.objects(frag, ONTO.hasActivation))
            for act in old_acts:
                g.remove((frag, ONTO.hasActivation, act))
                g.remove((frag, ONTO.hasPrimaryActivation, act))
                g.remove((frag, ONTO.hasSecondaryActivation, act))
                # remove todas as triplas onde a ativação é sujeito
                g.remove((act, None, None))
            # também limpa ligações que possam ter sobrado
            g.remove((frag, ONTO.hasPrimaryActivation, None))
            g.remove((frag, ONTO.hasSecondaryActivation, None))

        # --- resolve inteligências e consolida por "localname" ---
        resolved: List[Tuple[str, URIRef, float]] = []  # (local, cls, val)
        for name, val in scores.items():
            if val is None:
                continue
            valf = float(val)
            if valf <= 0.0:
                continue

            key_lc = str(name).strip().lower()

            cls = intel_index.get(key_lc)
            if cls is None:
                key_norm = key_lc.replace("-", " ").replace("_", " ")
                cls = intel_index.get(key_norm)
            if cls is None:
                continue  # não resolve => ignora

            local = str(cls).rsplit("#", 1)[-1]
            resolved.append((local, cls, valf))

        if not resolved:
            continue

        # consolida caso haja duplicatas da mesma inteligência
        per_local: Dict[str, Tuple[URIRef, float]] = {}
        for local, cls, valf in resolved:
            if local in per_local:
                prev_cls, prev_val = per_local[local]
                per_local[local] = (prev_cls, prev_val + valf)  # soma (ou troque por max, se preferir)
            else:
                per_local[local] = (cls, valf)

        items = [(local, cls, valf) for local, (cls, valf) in per_local.items()]
        max_score = max(valf for _, _, valf in items)
        if max_score <= 0.0:
            continue

        sum_score = sum(valf for _, _, valf in items)
        if sum_score <= 0.0:
            continue

        # define primária (1 só). Tie-break determinístico: maior score, depois local
        items_sorted = sorted(items, key=lambda t: (-t[2], t[0]))
        primary_local = items_sorted[0][0]

        secondary_threshold = theta * max_score
        base_ns = _base_from_fragment_uri(frag)
        created_map[k] = []

        for local, cls, valf in items_sorted:
            act_uri = URIRef(f"{base_ns}act-{k:03d}-{local}")

            # Tipagem base
            g.add((act_uri, RDF.type, ONTO.IntelligenceActivation))
            valf_norm = valf / sum_score

            g.set((act_uri, ONTO.hasActivationScore,
                Literal(_qdec(valf_norm, vec_places), datatype=XSD.decimal)))

            # opcional (recomendado): guarda o bruto para auditoria/debug
            g.set((act_uri, ONTO.hasWeight,
                Literal(_qdec(valf, vec_places), datatype=XSD.decimal)))

            # is about -> IntelligenceDisposition (classe)
            g.set((act_uri, IAO["0000136"], cls))

            if link_isAboutFragment:
                g.set((act_uri, ONTO.isAboutFragment, frag))

            # Liga sempre em hasActivation
            g.add((frag, ONTO.hasActivation, act_uri))

            is_primary = (local == primary_local)

            # IMPORTANTÍSSIMO: 1 único hasActivationType por ativação
            g.set((act_uri, ONTO.hasActivationType, ONTO.Primary if is_primary else ONTO.Secondary))

            if is_primary:
                # IMPORTANTÍSSIMO: 1 único hasPrimaryActivation por fragmento
                g.set((frag, ONTO.hasPrimaryActivation, act_uri))
            else:
                if valf >= secondary_threshold:
                    g.add((frag, ONTO.hasSecondaryActivation, act_uri))

            created_map[k].append(act_uri)

    return g, created_map


# ------------------------------------------------------------
# S4 — compute a_{k,j} a partir do evidences_payload.json (preferível)
# ------------------------------------------------------------
def compute_scores_by_fragment_from_payload(
    g: Graph,
    fragment_nodes: List[URIRef],
    evidences_payload: Dict[int, List[Dict[str, Any]]],
    *,
    w_keyword: float = 1.00,
    w_context_object: float = 1.25,
    w_discursive_strategy: float = 1.10,
    default_relevance: float = 1.0,
    default_confidence: float = 1.0,
) -> Dict[int, Dict[str, float]]:
    """
    Calcula scores_by_fragment[k][intel_localname] = a_{k,j} usando o JSON do S3.

    Fórmula (conforme seu texto):
      a_{k,j} = sum_{e in E_{k,j}} ( w_role(e) * r(e) * c(e) )

    Onde:
      - w_role: depende de role (Keyword/ContextObject/DiscursiveStrategy)
      - r(e)=relevance (se não existir -> default_relevance)
      - c(e)=confidence (se não existir -> default_confidence)

    Observação:
      - A resolução de "intelligence" é feita contra o grafo (ontologia + instâncias).
      - Se não resolver, ignora (não inventa nada).
    """
    intel_index = _index_intelligences(g)

    role_key_to_weight = {
        "keyword": float(w_keyword),
        "contextobject": float(w_context_object),
        "discursivestrategy": float(w_discursive_strategy),
    }

    scores_by_fragment: Dict[int, Dict[str, float]] = {}

    for k, _frag in enumerate(fragment_nodes):
        evs = evidences_payload.get(k) or []
        if not evs:
            continue

        per_intel: Dict[str, float] = {}

        for ev in evs:
            role_raw = (ev.get("role") or "").strip()
            intel_raw = (ev.get("intelligence") or "").strip()
            if not role_raw or not intel_raw:
                continue

            role_key = role_raw.replace("_", "").replace("-", "").lower()
            w_role = role_key_to_weight.get(role_key)
            if w_role is None:
                continue

            r = ev.get("relevance", default_relevance)
            c = ev.get("confidence", default_confidence)
            try:
                r = float(r)
            except Exception:
                r = float(default_relevance)
            try:
                c = float(c)
            except Exception:
                c = float(default_confidence)

            intel_name = intel_raw.strip().lower()

            cls = intel_index.get(intel_name)
            if cls is None:
                # variações simples
                variants = {
                    intel_name,
                    intel_name.replace("-", " ").replace("_", " "),
                }
                for v in variants:
                    if v in intel_index:
                        cls = intel_index[v]
                        break
            if cls is None:
                continue

            local = str(cls).rsplit("#", 1)[-1]  # localname estável
            per_intel[local] = per_intel.get(local, 0.0) + (w_role * r * c)

        if per_intel:
            scores_by_fragment[k] = per_intel

    return scores_by_fragment


# ------------------------------------------------------------
# S4 — Runner por arquivos (para o run_batch)
# ------------------------------------------------------------
def run_s4_from_files(
    *,
    s3_ttl_path: Path,
    out_ttl_path: Path,
    onto_path: Optional[Path] = None,
    evidences_payload_json_path: Optional[Path] = None,
    scores_json_path: Optional[Path] = None,
    theta: float = 0.75,
    w_keyword: float = 1.00,
    w_context_object: float = 1.25,
    w_discursive_strategy: float = 1.10,
    link_isAboutFragment: bool = True,
    vec_places: int = 4,
) -> Dict[str, Any]:
    """
    Executa S4 a partir do TTL do S3, gerando:
      - instances_fragments_activations.ttl
      - (opcional) scores_by_fragment.json

    Preferência de cálculo:
      1) Se evidences_payload_json_path existir: calcula a_{k,j} usando JSON (w_role*r*c).
      2) Caso contrário: fallback no grafo (somente w_role, via usesElement/evokesIntelligence).
    """
    if not s3_ttl_path.exists():
        raise FileNotFoundError(f"S3 TTL não encontrado: {s3_ttl_path}")

    # Carrega: ontologia (opcional) + instâncias S3
    g = Graph()
    if onto_path is not None and onto_path.exists():
        g.parse(onto_path.as_posix(), format="turtle")
    g.parse(s3_ttl_path.as_posix(), format="turtle")

    fragment_nodes = sorted(
        (s for s, _, _ in g.triples((None, RDF.type, ONTO.ExplanationFragment))),
        key=lambda u: str(u),
    )

    payload_obj: Optional[Dict[int, List[Dict[str, Any]]]] = None
    if evidences_payload_json_path is not None and evidences_payload_json_path.exists():
        raw = json.loads(evidences_payload_json_path.read_text(encoding="utf-8"))
        # JSON pode vir com chaves string; normaliza para int
        payload_obj = {int(k): v for k, v in raw.items()}

    if payload_obj is not None:
        scores_by_fragment = compute_scores_by_fragment_from_payload(
            g,
            fragment_nodes,
            payload_obj,
            w_keyword=w_keyword,
            w_context_object=w_context_object,
            w_discursive_strategy=w_discursive_strategy,
        )
        scores_source = "payload_json"
    else:
        scores_by_fragment = compute_scores_by_fragment_from_graph(
            g,
            fragment_nodes,
            w_keyword=w_keyword,
            w_context_object=w_context_object,
            w_discursive_strategy=w_discursive_strategy,
        )
        scores_source = "graph_fallback"

    if scores_json_path is not None:
        scores_json_path.write_text(
            json.dumps(scores_by_fragment, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Cria ativações
    g, created = add_activations_rdf(
        g,
        fragment_nodes,
        scores_by_fragment,
        theta=theta,
        link_isAboutFragment=link_isAboutFragment,
        vec_places=vec_places,
    )

    g.serialize(destination=out_ttl_path.as_posix(), format="turtle")

    # métricas
    activations_total = sum(len(v) for v in created.values())
    primary_total = len(created)  # por fragmento com scores -> 1 primária
    secondary_total = max(0, activations_total - primary_total)

    secondary_linked_total = 0
    for frag in fragment_nodes:
        for _act in g.objects(frag, ONTO.hasSecondaryActivation):
            secondary_linked_total += 1

    return {
        "status": "ok",
        "input_s3_ttl": s3_ttl_path.name,
        "output_ttl": out_ttl_path.name,
        "scores_source": scores_source,
        "theta": theta,
        "weights": {
            "keyword": w_keyword,
            "context_object": w_context_object,
            "discursive_strategy": w_discursive_strategy,
        },
        "fragments_total": len(fragment_nodes),
        "fragments_with_scores": len(created),
        "activations_total": activations_total,
        "primary_total": primary_total,
        "secondary_total": secondary_total,
        "secondary_linked_total": secondary_linked_total,
        "scores_json": (scores_json_path.name if scores_json_path is not None else None),
        "payload_json": (evidences_payload_json_path.name if evidences_payload_json_path is not None else None),
    }


# ------------------------------------------------------------
# CLI de teste mínimo
# ------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument("--in", dest="in_ttl", default="instances_fragments_evidences.ttl")
    ap.add_argument("--out", dest="out_ttl", default="instances_fragments_activations.ttl")
    ap.add_argument("--theta", type=float, default=0.75)
    ap.add_argument("--w-keyword", type=float, default=1.00)
    ap.add_argument("--w-context", type=float, default=1.25)
    ap.add_argument("--w-strategy", type=float, default=1.10)
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_ttl, format="turtle")

    fragment_nodes = sorted(
        (s for s, _, _ in g.triples((None, RDF.type, ONTO.ExplanationFragment))),
        key=lambda u: str(u)
    )

    scores_by_fragment = compute_scores_by_fragment_from_graph(
        g,
        fragment_nodes,
        w_keyword=args.w_keyword,
        w_context_object=args.w_context,
        w_discursive_strategy=args.w_strategy,
    )

    g, acts = add_activations_rdf(g, fragment_nodes, scores_by_fragment, theta=args.theta)

    g.serialize(destination=args.out_ttl, format="turtle")
    print("Ativações criadas por fragmento:")
    for k, uris in acts.items():
        print(f"[frag {k:03d}]")
        for u in uris:
            print(" -", u)
    print("Grafo salvo em:", args.out_ttl)

