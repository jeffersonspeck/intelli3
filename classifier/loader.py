
"""
PT-BR: Loader de vetores MIProfile (ou MIProfileVector) a partir de arquivos TTL.
EN: MIProfile (or MIProfileVector) vector loader from TTL files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from rdflib import Graph, Namespace, RDF, Literal, URIRef

# Namespaces mais comuns do seu projeto.
ONTO_CANDIDATES = [
    Namespace("https://techcoop.com.br/ontomi#"),
    Namespace("http://purl.org/ontomi/onto#"),
]

# Classes possíveis (você usa MIProfileVector no TTL atual)
MI_CLASSES_LOCALNAMES = {"MIProfile", "MIProfileVector"}

# Propriedades prováveis para vetor serializado
VEC_PROPS_LOCALNAMES = {"miVector", "hasMIVector", "hasProfileVector"}

# Propriedades por-dimensão (fallback se não houver miVector literal)
SCORE_PROPS_LOCALNAMES = [
    "hasLinguisticScore",
    "hasLogicalMathematicalScore",
    "hasSpatialScore",
    "hasMusicalScore",
    "hasBodilyKinestheticScore",
    "hasInterpersonalScore",
    "hasIntrapersonalScore",
    "hasNaturalistScore",
]

@dataclass(frozen=True)
class LoadedDoc:
    doc_id: str
    vector: List[float]
    source_path: str

def _localname(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]

def _parse_vector_literal(lit: str) -> Optional[List[float]]:
    if not lit:
        return None
    raw = lit.strip().strip("[]()")
    parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    try:
        vec = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vec) != 8:
        return None
    return vec

def _vec_from_score_props(g: Graph, subj: URIRef, onto_ns: Namespace) -> Optional[List[float]]:
    vals: List[float] = []
    for ln in SCORE_PROPS_LOCALNAMES:
        prop = getattr(onto_ns, ln, None)
        if prop is None:
            return None
        obj = next(g.objects(subj, prop), None)
        if obj is None:
            return None
        try:
            vals.append(float(obj))
        except Exception:
            return None
    return vals if len(vals) == 8 else None

def load_one_ttl(path: str) -> Optional[LoadedDoc]:
    p = Path(path)
    if not p.exists():
        return None

    g = Graph()
    try:
        g.parse(str(p), format="turtle")
    except Exception:
        g.parse(str(p))

    # Descobre sujeitos candidatos por rdf:type cujo localname é MIProfileVector/MIProfile
    candidates: List[URIRef] = []
    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef) and _localname(o) in MI_CLASSES_LOCALNAMES:
            candidates.append(s)

    if not candidates:
        return None

    # Tenta cada namespace conhecido (e também tenta inferir pelo próprio grafo)
    namespaces_to_try = list(ONTO_CANDIDATES)
    # adiciona namespaces que aparecem nas URIs de tipo
    for _, _, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef) and ("#" in str(o)):
            ns = str(o).rsplit("#", 1)[0] + "#"
            namespaces_to_try.append(Namespace(ns))

    # remove duplicados mantendo ordem
    seen=set()
    uniq=[]
    for ns in namespaces_to_try:
        s=str(ns)
        if s not in seen:
            uniq.append(ns); seen.add(s)
    namespaces_to_try=uniq

    for subj in candidates:
        doc_id = str(subj)

        # 1) tenta vetor serializado (miVector)
        for ns in namespaces_to_try:
            for prop_ln in VEC_PROPS_LOCALNAMES:
                prop = getattr(ns, prop_ln, None)
                if prop is None:
                    continue
                for obj in g.objects(subj, prop):
                    if isinstance(obj, Literal):
                        vec = _parse_vector_literal(str(obj))
                        if vec:
                            return LoadedDoc(doc_id=doc_id, vector=vec, source_path=str(p))

        # 2) fallback: monta vetor pelas 8 propriedades has*Score
        for ns in namespaces_to_try:
            vec = _vec_from_score_props(g, subj, ns)
            if vec:
                return LoadedDoc(doc_id=doc_id, vector=vec, source_path=str(p))

    return None

def load_from_output_dir(output_dir: str, *, filename: str = "instances_fragments_profile.ttl") -> List[LoadedDoc]:
    root = Path(output_dir)
    if not root.exists():
        return []
    out: List[LoadedDoc] = []
    for p in root.rglob(filename):
        doc = load_one_ttl(str(p))
        if doc:
            out.append(doc)
    return out
