"""\
PT-BR: S5 — agrega ativações e gera vetores de perfil MI por documento.
EN: S5 — aggregates activations and builds MI profile vectors per document.
"""

# ------------------------------------------------------------
# Intelli3 / OntoMI — S5: Vetor de Perfil (v_MI) por documento
#
# Este script implementa a agregação e normalização descritas na Seção
# "Algoritmo de classificação e quantificação" (Equações de doc-score e norm):
#
# - Assume que o grafo já contém, para cada ExplanationFragment k, ativações
#   (a_{k,j}) instanciadas como:
#       frag --onto:hasActivation--> act
#       act  --iao:0000136--> IntelligenceDispositionClass
#       act  --onto:hasActivationScore--> decimal
#
# - Agrega no nível do documento:
#       s_j = sum_k alpha_k * a_{k,j}
#
# - Normaliza para obter v_MI usando: l1 (padrão), l2 ou softmax_tau.
#
# OBS: os termos w_role, r(e), c(e) (Eq. de fragment-score) são tratados
# upstream (ex.: no cálculo de a_{k,j}); aqui apenas consolidamos e normalizamos.
# ------------------------------------------------------------
from __future__ import annotations

import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Literal

from rdflib import Graph, Namespace, URIRef, RDF, Literal as RDFLiteral, XSD

BFO   = Namespace("http://purl.obolibrary.org/obo/BFO_")
IAO   = Namespace("http://purl.obolibrary.org/obo/IAO_")
ONTO  = Namespace("https://techcoop.com.br/ontomi#")
OWL   = Namespace("http://www.w3.org/2002/07/owl#")
RDFNS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFSNS= Namespace("http://www.w3.org/2000/01/rdf-schema#")


# ------------------------------------------------------------
# Convenção do vetor v_MI (conforme texto da dissertação)
# (LING, LOG, ESP, MUS, CORP, INTER, INTRA, NAT)
# ------------------------------------------------------------
INTEL_ORDER: List[URIRef] = [
    ONTO.LinguisticIntelligence,
    ONTO.LogicalMathematicalIntelligence,
    ONTO.SpatialIntelligence,
    ONTO.MusicalIntelligence,
    ONTO.BodilyKinestheticIntelligence,
    ONTO.InterpersonalIntelligence,
    ONTO.IntrapersonalIntelligence,
    ONTO.NaturalistIntelligence,
]

# Propriedades correspondentes (mesma ordem)
INTEL_PROPS: List[URIRef] = [
    ONTO.hasLinguisticScore,
    ONTO.hasLogicalMathematicalScore,
    ONTO.hasSpatialScore,
    ONTO.hasMusicalScore,
    ONTO.hasBodilyKinestheticScore,
    ONTO.hasInterpersonalScore,
    ONTO.hasIntrapersonalScore,
    ONTO.hasNaturalistScore,
]

NormType = Literal["l1", "l2", "softmax"]


# ------------------------------------------------------------
# Helpers numéricos / URIs
# ------------------------------------------------------------
def _qdec(x: Decimal, places: int = 4) -> Decimal:
    # Eu uso quantização para manter os números padronizados.
    return x.quantize(Decimal("1." + "0" * places), rounding=ROUND_HALF_UP)

def _base_from_uri(u: URIRef) -> str:
    s = str(u)
    if "#" in s:
        return s.split("#", 1)[0] + "#"
    return s.rstrip("/") + "#"

def _doc_uri_from_base(base: str) -> URIRef:
    # base: "urn:doc:xyz#" -> doc URI "urn:doc:xyz"
    if base.endswith("#"):
        return URIRef(base[:-1])
    return URIRef(base.rstrip("/"))

def _extract_k_from_fragment_uri(frag: URIRef) -> Optional[int]:
    m = re.search(r"frag-(\d+)", str(frag))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _collect_fragment_nodes(g: Graph) -> List[URIRef]:
    # Aqui eu ordeno por URI para manter determinismo no output.
    return sorted(
        (s for s, _, _ in g.triples((None, RDF.type, ONTO.ExplanationFragment))),
        key=lambda u: str(u),
    )

def _sum_scores_by_intel_for_fragment(g: Graph, frag: URIRef) -> Dict[URIRef, Decimal]:
    """
    Soma scores por classe de inteligência (a_{k,j}) para um fragmento.
    Espera:
      frag --onto:hasActivation--> act
      act  --iao:0000136--> IntelligenceDispositionClass
      act  --onto:hasActivationScore--> decimal
    """
    totals: Dict[URIRef, Decimal] = {}
    for _, _, act in g.triples((frag, ONTO.hasActivation, None)):
        # classe alvo (IntelligenceDispositionClass)
        target_cls: Optional[URIRef] = None
        for _, _, cls in g.triples((act, IAO["0000136"], None)):
            target_cls = cls  # pega o primeiro
            break
        if target_cls is None:
            continue

        score_val: Optional[Decimal] = None
        for _, _, lit in g.triples((act, ONTO.hasActivationScore, None)):
            try:
                score_val = Decimal(str(lit))
            except Exception:
                score_val = None
            break
        if score_val is None:
            continue

        totals[target_cls] = totals.get(target_cls, Decimal("0")) + score_val

    return totals


def _normalize(vec: List[Decimal], norm_type: NormType = "l1", tau: float = 1.0) -> List[Decimal]:
    # Se tudo zero, mantém tudo zero (sanidade / reprodutibilidade)
    if all(v == 0 for v in vec):
        return [Decimal("0") for _ in vec]

    if norm_type == "l1":
        denom = sum(vec, Decimal("0"))
        if denom == 0:
            return [Decimal("0") for _ in vec]
        return [v / denom for v in vec]

    if norm_type == "l2":
        # usa float para sqrt, retorna Decimal
        sq = sum((float(v) ** 2 for v in vec))
        if sq <= 0.0:
            return [Decimal("0") for _ in vec]
        denom = math.sqrt(sq)
        return [Decimal(str(float(v) / denom)) for v in vec]

    # softmax
    if tau <= 0:
        tau = 1.0
    xs = [float(v) / tau for v in vec]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]  # estabilidade numérica
    denom = sum(exps)
    if denom <= 0.0:
        return [Decimal("0") for _ in vec]
    return [Decimal(str(e / denom)) for e in exps]


def _format_percent_string(vec_norm: List[Decimal]) -> str:
    # Percentuais com duas casas, separados por vírgula, na ordem do vetor canônico
    return ",".join(f"{(v * Decimal('100')):.2f}" for v in vec_norm)


# ------------------------------------------------------------
# Leitura opcional de metadados do S1 para alpha_k (título/resumo vs corpo)
# ------------------------------------------------------------
def _load_alpha_map_from_s1_json(s1_json_path: str) -> Dict[int, bool]:
    """
    Retorna dict[k] = is_title_or_abstract (bool), se disponível.
    Formato esperado:
      { "doc_id": "...", "paragraphs": [ {"k":0,"text":"...","is_title_or_abstract": true|false, ...}, ...] }
    """
    with open(s1_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    alpha_map: Dict[int, bool] = {}
    for p in (doc.get("paragraphs") or []):
        k = p.get("k")
        if k is None:
            continue
        ita = p.get("is_title_or_abstract")
        if isinstance(ita, bool):
            alpha_map[int(k)] = ita
    return alpha_map


# ------------------------------------------------------------
# Construção dos vetores
# ------------------------------------------------------------
def build_document_profile_vector(
    g: Graph,
    *,
    s1_json_path: Optional[str] = None,
    alpha_title_abstract: float = 1.30,
    alpha_body: float = 1.00,
    norm_type: NormType = "l1",
    tau: float = 1.0,
    vec_places: int = 4,
    write_percent_string: bool = True,
) -> Tuple[Graph, URIRef, URIRef, List[Decimal]]:
    """
    Constrói 1 vetor v_MI para o documento (agregando todos os fragments do grafo).

    Retorna:
      (g, doc_uri, vec_uri, vec_norm_list[8])
    """
    frags = _collect_fragment_nodes(g)
    if not frags:
        raise ValueError("Nenhum ExplanationFragment encontrado no grafo.")

    base = _base_from_uri(frags[0])
    doc_uri = _doc_uri_from_base(base)
    vec_uri = URIRef(f"{base}vec-doc")

    alpha_map: Dict[int, bool] = {}
    if s1_json_path:
        try:
            alpha_map = _load_alpha_map_from_s1_json(s1_json_path)
        except Exception:
            alpha_map = {}

    # s_j (bruto por inteligência, no eixo canônico)
    s_raw = [Decimal("0") for _ in INTEL_ORDER]

    for frag in frags:
        totals = _sum_scores_by_intel_for_fragment(g, frag)  # a_{k,j} por classe
        k = _extract_k_from_fragment_uri(frag)
        is_title_abs = bool(alpha_map.get(k, False)) if k is not None else False
        alpha_k = Decimal(str(alpha_title_abstract if is_title_abs else alpha_body))

        for j, cls in enumerate(INTEL_ORDER):
            s_raw[j] += alpha_k * totals.get(cls, Decimal("0"))

    v_norm = _normalize(s_raw, norm_type=norm_type, tau=tau)

    # cria MIProfileVector e escreve propriedades
    g.add((vec_uri, RDF.type, ONTO.MIProfileVector))
    for p, v in zip(INTEL_PROPS, v_norm):
        g.add((vec_uri, p, RDFLiteral(_qdec(v, vec_places), datatype=XSD.decimal)))

    if write_percent_string:
        g.add((vec_uri, ONTO.miVector, RDFLiteral(_format_percent_string(v_norm), datatype=XSD.string)))

    # liga ao documento (nó do doc)
    g.add((doc_uri, ONTO.hasProfileVector, vec_uri))

    return g, doc_uri, vec_uri, v_norm


def build_fragment_profile_vectors(
    g: Graph,
    *,
    norm_type: NormType = "l1",
    tau: float = 1.0,
    vec_places: int = 4,
    write_percent_string: bool = True,
) -> Tuple[Graph, Dict[URIRef, URIRef]]:
    """
    (Opcional) Cria um MIProfileVector para cada ExplanationFragment, usando
    apenas as ativações daquele fragmento (a_{k,j}), e liga via onto:hasProfileVector.

    Útil para inspeção/localidade e compatibilidade com validações existentes.
    """
    mapping: Dict[URIRef, URIRef] = {}
    frags = _collect_fragment_nodes(g)

    for frag in frags:
        totals = _sum_scores_by_intel_for_fragment(g, frag)
        raw = [totals.get(cls, Decimal("0")) for cls in INTEL_ORDER]
        v_norm = _normalize(raw, norm_type=norm_type, tau=tau)

        base = _base_from_uri(frag)
        frag_id = str(frag).rsplit("#", 1)[-1]
        vec_uri = URIRef(f"{base}vec-{frag_id}")

        g.add((vec_uri, RDF.type, ONTO.MIProfileVector))
        for p, v in zip(INTEL_PROPS, v_norm):
            g.add((vec_uri, p, RDFLiteral(_qdec(v, vec_places), datatype=XSD.decimal)))

        if write_percent_string:
            g.add((vec_uri, ONTO.miVector, RDFLiteral(_format_percent_string(v_norm), datatype=XSD.string)))

        g.add((frag, ONTO.hasProfileVector, vec_uri))
        mapping[frag] = vec_uri

    return g, mapping

def run_s5_from_files(
    *,
    in_ttl_path: str,
    out_ttl_path: str,
    s1_json_path: Optional[str] = None,
    scope: str = "both",  # "document" | "fragment" | "both"
    norm: NormType = "l1",
    tau: float = 1.0,
    alpha_title: float = 1.30,
    alpha_body: float = 1.00,
    vec_places: int = 4,
    write_percent_string: bool = True,
) -> Dict[str, object]:
    """
    Runner amigável para o run_batch:
      - carrega TTL (após S4)
      - gera vetores (documento e/ou fragmento)
      - salva TTL final (profile)
      - retorna métricas p/ log
    """
    g = Graph()
    g.parse(in_ttl_path, format="turtle")

    metrics: Dict[str, object] = {
        "scope": scope,
        "norm": norm,
        "tau": float(tau),
        "alpha_title": float(alpha_title),
        "alpha_body": float(alpha_body),
        "vec_places": int(vec_places),
        "write_miVector_string": bool(write_percent_string),
    }

    if scope in ("fragment", "both"):
        g, frag_map = build_fragment_profile_vectors(
            g,
            norm_type=norm,
            tau=tau,
            vec_places=vec_places,
            write_percent_string=write_percent_string,
        )
        metrics["fragment_vectors"] = len(frag_map)

    doc_uri = None
    vec_uri = None
    mi_vec_pct = None

    if scope in ("document", "both"):
        g, doc_uri, vec_uri, v_norm = build_document_profile_vector(
            g,
            s1_json_path=s1_json_path,
            alpha_title_abstract=alpha_title,
            alpha_body=alpha_body,
            norm_type=norm,
            tau=tau,
            vec_places=vec_places,
            write_percent_string=write_percent_string,
        )
        metrics["doc_uri"] = str(doc_uri)
        metrics["doc_vector_uri"] = str(vec_uri)

        if write_percent_string:
            mi_vec_pct = _format_percent_string(v_norm)
            metrics["miVector_pct"] = mi_vec_pct

    g.serialize(destination=out_ttl_path, format="turtle")
    metrics["output"] = out_ttl_path

    return metrics

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="S5 — gera vetor MIProfileVector por documento e/ou por fragmento.")
    ap.add_argument("--in", dest="in_path", default="instances_fragments_activations.ttl",
                    help="TTL de entrada (após S4).")
    ap.add_argument("--out", dest="out_path", default="instances_fragments_profile.ttl",
                    help="TTL de saída.")
    ap.add_argument("--s1-json", dest="s1_json_path", default=None,
                    help="(Opcional) Caminho do s1_output.json para aplicar alpha_k (título/resumo vs corpo).")
    ap.add_argument("--scope", choices=["document", "fragment", "both"], default="both",
                    help="Gera vetor por documento, por fragmento, ou ambos (default: both).")
    ap.add_argument("--norm", choices=["l1", "l2", "softmax"], default="l1",
                    help="Normalização do vetor (default: l1).")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Temperatura do softmax (apenas se --norm=softmax).")
    ap.add_argument("--alpha-title", type=float, default=1.30,
                    help="alpha_k para fragmentos de título/resumo (default: 1.30).")
    ap.add_argument("--alpha-body", type=float, default=1.00,
                    help="alpha_k para fragmentos do corpo (default: 1.00).")
    ap.add_argument("--vec-places", type=int, default=4,
                    help="Casas decimais nos scores (default: 4).")
    ap.add_argument("--no-mi-vector-string", action="store_true",
                    help="Não escreve a propriedade onto:miVector com percentuais.")

    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_path, format="turtle")

    write_str = not args.no_mi_vector_string
    norm: NormType = args.norm  # type: ignore[assignment]

    if args.scope in ("fragment", "both"):
        g, frag_map = build_fragment_profile_vectors(
            g,
            norm_type=norm,
            tau=args.tau,
            vec_places=args.vec_places,
            write_percent_string=write_str,
        )
        print(f"[OK] Vetores por fragmento: {len(frag_map)}")

    doc_vec_uri = None
    doc_uri = None
    doc_vec = None
    if args.scope in ("document", "both"):
        g, doc_uri, doc_vec_uri, doc_vec = build_document_profile_vector(
            g,
            s1_json_path=args.s1_json_path,
            alpha_title_abstract=args.alpha_title,
            alpha_body=args.alpha_body,
            norm_type=norm,
            tau=args.tau,
            vec_places=args.vec_places,
            write_percent_string=write_str,
        )
        if doc_vec is not None:
            print("[OK] Vetor do documento:")
            print(" - doc:", doc_uri)
            print(" - vec:", doc_vec_uri)
            if write_str:
                print(" - miVector (%):", _format_percent_string(doc_vec))

    g.serialize(destination=args.out_path, format="turtle")
    print(f"[OK] Grafo salvo em: {args.out_path}")
