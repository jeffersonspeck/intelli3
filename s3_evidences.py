# s3_evidences.py
# ------------------------------------------------------------
# Intelli3 / OntoMI — S3: Instanciação de Evidências (RDF)
#
# - Lê S2 (TTL) e acha os fragments (ordem estável)
# - Gera payload por fragmento via evidences_api (LLM opcional)
# - Instancia Keyword/ContextObject/DiscursiveStrategy
# - Liga frag -> usesElement -> evid
# - Opcional: evid -> evokesIntelligence -> IntelligenceDispositionClass
#
# Também salva:
# - evidences_payload.json (o payload cru por k)
# - instances_fragments_evidences.ttl (grafo atualizado)
# ------------------------------------------------------------

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

import evidences_api as ea

annotate_with_llm_default = ea.annotate_with_llm_default
start_ollama_daemon_if_needed = ea.start_ollama_daemon_if_needed
ensure_model_available = ea.ensure_model_available
wait_ollama_ready = ea.wait_ollama_ready

DEFAULT_MODEL = getattr(ea, "DEFAULT_MODEL", "mistral:7b-instruct")
OLLAMA_HOST  = getattr(ea, "OLLAMA_HOST",  "http://127.0.0.1:11434")


def make_payload_from_s1_json_file(s1_json_path: str, *, annotate_fn):
    """
    Compat layer:
    - Se evidences_api tiver make_payload_from_s1_json_file, usa.
    - Senão, lê o S1 e executa annotate_fn(doc) direto.
    Retorna {k(int): [evidences...]}.
    """
    fn = getattr(ea, "make_payload_from_s1_json_file", None)
    if callable(fn):
        try:
            out = fn(s1_json_path, annotate_fn=annotate_fn)
        except TypeError:
            # caso a assinatura antiga seja (path, annotate_fn)
            out = fn(s1_json_path, annotate_fn)
    else:
        doc = json.loads(Path(s1_json_path).read_text(encoding="utf-8"))
        out = annotate_fn(doc)

    # garante chaves int
    fixed = {}
    for k, v in (out or {}).items():
        try:
            fixed[int(k)] = v
        except Exception:
            # se vier algo bizarro, mantém como está
            fixed[k] = v
    return fixed

# Namespaces exatamente como na OntoMI
BFO  = Namespace("http://purl.obolibrary.org/obo/BFO_")
IAO  = Namespace("http://purl.obolibrary.org/obo/IAO_")
ONTO = Namespace("https://techcoop.com.br/ontomi#")
OWL  = Namespace("http://www.w3.org/2002/07/owl#")
DCT  = Namespace("http://purl.org/dc/terms/")

# cache para não ficar “subindo” ollama por doc
_OLLAMA_READY = False

# Mapeia role -> classe OntoMI (não inventa fora)
ROLE_TO_CLASS = {
    "keyword": ONTO.Keyword,
    "contextobject": ONTO.ContextObject,
    "discursivestrategy": ONTO.DiscursiveStrategy,
}

def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _base_from_fragment_uri(frag_uri: URIRef) -> str:
    u = str(frag_uri)
    if "#" in u:
        return u.split("#", 1)[0] + "#"
    return u.rstrip("/") + "#"

def _slug(text: str, maxlen: int = 32) -> str:
    s = re.sub(r"\s+", "-", (text or "").strip().lower())
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    return s[:maxlen] if len(s) > maxlen else s

def _collect_fragment_nodes(g: Graph) -> List[URIRef]:
    """
    Recupera fragments em ordem estável.
    Prioriza dct:identifier (se existir, vindo da S2), senão ordena por URI.
    """
    frags = [s for s, _, _ in g.triples((None, RDF.type, ONTO.ExplanationFragment))]
    if not frags:
        return []

    # tenta ordenar por DCT.identifier inteiro
    scored: List[Tuple[int, URIRef]] = []
    for f in frags:
        ident = None
        for _, _, lit in g.triples((f, DCT.identifier, None)):
            try:
                ident = int(str(lit))
                break
            except Exception:
                ident = None
        if ident is None:
            # fallback: tenta extrair ...frag-XYZ do URI
            m = re.search(r"frag-(\d+)", str(f))
            ident = int(m.group(1)) if m else 10**9
        scored.append((ident, f))

    scored.sort(key=lambda t: (t[0], str(t[1])))
    return [f for _, f in scored]

def _index_intelligences_from_graph(g: Graph) -> Dict[str, URIRef]:
    """
    Indexa subclasses de onto:IntelligenceDisposition (por localname e rdfs:label).
    """
    idx: Dict[str, URIRef] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        if (cls, RDFS.subClassOf, ONTO.IntelligenceDisposition) in g:
            local = str(cls).rsplit("#", 1)[-1].lower()
            idx[local] = cls
            for _, _, lbl in g.triples((cls, RDFS.label, None)):
                idx[str(lbl).strip().lower()] = cls
    return idx

def _build_intelligence_index(g_instances: Graph, onto_path: Optional[Path]) -> Dict[str, URIRef]:
    """
    Se o grafo já contém a ontologia (modo full), usa ele.
    Se não contém (modo instances), parseia a ontologia só para indexar.
    """
    idx = _index_intelligences_from_graph(g_instances)
    if idx:
        return idx

    if onto_path and onto_path.exists():
        g_onto = Graph()
        g_onto.parse(str(onto_path), format="turtle")
        return _index_intelligences_from_graph(g_onto)

    return {}

def _resolve_intelligence(intel_index: Dict[str, URIRef], name: str) -> Optional[URIRef]:
    if not name:
        return None
    key = name.strip().lower()
    if key in intel_index:
        return intel_index[key]
    variants = {
        key.replace("-", " ").replace("_", " "),
        key.replace("  ", " "),
    }
    for v in variants:
        if v in intel_index:
            return intel_index[v]
    return None

def _safe_annotate_fn(use_llm: bool) -> Tuple[callable, bool]:
    """
    Retorna (annotate_fn, will_use_llm).
    Se a versão instalada não suportar toggle via use_llm, assume que usa LLM sempre.
    """
    sig = inspect.signature(annotate_with_llm_default)
    supports_toggle = "use_llm" in sig.parameters

    if supports_toggle:
        annotate_fn = annotate_with_llm_default(use_llm=use_llm)
        return annotate_fn, use_llm

    # não suporta desligar -> assume LLM sempre
    annotate_fn = annotate_with_llm_default()
    return annotate_fn, True

def ensure_ollama_ready_if_needed(will_use_llm: bool) -> None:
    global _OLLAMA_READY
    if not will_use_llm:
        return
    if _OLLAMA_READY:
        return

    start_ollama_daemon_if_needed(OLLAMA_HOST)
    ensure_model_available(DEFAULT_MODEL)
    wait_ollama_ready(OLLAMA_HOST, DEFAULT_MODEL)
    _OLLAMA_READY = True

def add_evidences_rdf(
    g: Graph,
    fragment_nodes: List[URIRef],
    evidences_payload: Dict[int, List[Dict[str, Any]]],
    *,
    onto_path: Optional[Path] = None,
    link_evokes_intelligence: bool = True,
) -> Tuple[Graph, Dict[int, List[URIRef]], Dict[str, int]]:
    """
    Instancia evidências e liga frag->usesElement->evid.

    Retorna:
      (g, created_map, role_counts)
    """
    intel_index = _build_intelligence_index(g, onto_path) if link_evokes_intelligence else {}
    created_map: Dict[int, List[URIRef]] = {}
    role_counts: Dict[str, int] = {"keyword": 0, "contextobject": 0, "discursivestrategy": 0}

    for k, frag in enumerate(fragment_nodes):
        if k not in evidences_payload:
            continue

        base_ns = _base_from_fragment_uri(frag)
        created_map[k] = []

        for i, ev in enumerate(evidences_payload[k]):
            role_raw = (ev.get("role") or "").strip()
            text = (ev.get("text") or "").strip()
            lang = (ev.get("lang") or "und").strip()

            if not role_raw or not text:
                continue

            role_key = role_raw.replace("_", "").replace("-", "").lower()
            cls = ROLE_TO_CLASS.get(role_key)
            if cls is None:
                raise ValueError(
                    f"Role inválido para evidência: {role_raw}. "
                    f"Use exatamente: Keyword | ContextObject | DiscursiveStrategy."
                )

            evid_uri = URIRef(f"{base_ns}evid-{k:03d}-{i:02d}-{_slug(text)}")

            g.add((evid_uri, RDF.type, cls))

            preview = text.replace("\n", " ").strip()
            if len(preview) > 160:
                preview = preview[:157] + "…"
            g.add((evid_uri, RDFS.label, Literal(preview, lang=lang)))

            g.add((frag, ONTO.usesElement, evid_uri))

            # opcional: evokesIntelligence
            if link_evokes_intelligence:
                intel_name = (ev.get("intelligence") or "").strip()
                target_cls = _resolve_intelligence(intel_index, intel_name)
                if target_cls is not None:
                    g.add((evid_uri, ONTO.evokesIntelligence, target_cls))

            created_map[k].append(evid_uri)
            if role_key in role_counts:
                role_counts[role_key] += 1

    return g, created_map, role_counts

def run_s3_from_files(
    *,
    s1_json_path: Path,
    s2_ttl_path: Path,
    out_ttl_path: Path,
    out_payload_json_path: Path,
    onto_path: Optional[Path] = None,
    use_llm: bool = True,
    link_evokes_intelligence: bool = True,
) -> Dict[str, Any]:
    """
    Runner de S3:
    - carrega TTL S2
    - monta payload pelo S1 via evidences_api
    - instancia evidências
    - salva TTL e payload
    - retorna métricas
    """
    g = Graph()
    g.parse(str(s2_ttl_path), format="turtle")

    fragment_nodes = _collect_fragment_nodes(g)
    if not fragment_nodes:
        raise RuntimeError(f"Nenhum ExplanationFragment encontrado em: {s2_ttl_path}")

    annotate_fn, will_use_llm = _safe_annotate_fn(use_llm)
    ensure_ollama_ready_if_needed(will_use_llm)

    evidences_payload = make_payload_from_s1_json_file(
        str(s1_json_path),
        annotate_fn=annotate_fn,
    )

    # salva payload “cru” para auditoria
    _write_json(out_payload_json_path, evidences_payload)

    g, created_map, role_counts = add_evidences_rdf(
        g,
        fragment_nodes,
        evidences_payload,
        onto_path=onto_path,
        link_evokes_intelligence=link_evokes_intelligence,
    )

    g.serialize(destination=str(out_ttl_path), format="turtle")

    evid_total = sum(len(v) for v in created_map.values())
    return {
        "fragments": len(fragment_nodes),
        "evidences_total": evid_total,
        "role_counts": role_counts,
        "payload_file": out_payload_json_path.name,
        "output": out_ttl_path.name,
        "use_llm": bool(will_use_llm),
        "link_evokes_intelligence": bool(link_evokes_intelligence),
    }
