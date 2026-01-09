"""\
PT-BR: S6 — consultas de validação (CQ1/CQ2/CQ3) sobre o grafo final.
EN: S6 — validation queries (CQ1/CQ2/CQ3) over the final graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS

ONTO = Namespace("https://techcoop.com.br/ontomi#")
IAO  = Namespace("http://purl.obolibrary.org/obo/IAO_")

VECTOR_ORDER = [
    ("Linguistic",            ONTO.hasLinguisticScore),
    ("Logical-Mathematical",  ONTO.hasLogicalMathematicalScore),
    ("Spatial",               ONTO.hasSpatialScore),
    ("Bodily-Kinesthetic",    ONTO.hasBodilyKinestheticScore),
    ("Musical",               ONTO.hasMusicalScore),
    ("Interpersonal",         ONTO.hasInterpersonalScore),
    ("Intrapersonal",         ONTO.hasIntrapersonalScore),
    ("Naturalist",            ONTO.hasNaturalistScore),
]

def _first_literal_lang(g: Graph, subj, pred, lang: str) -> Optional[str]:
    for lit in g.objects(subj, pred):
        if getattr(lit, "language", None) == lang:
            return str(lit)
    return None

def _doc_label(g: Graph, doc) -> str:
    # Eu priorizo rótulos em pt/en para os relatórios ficarem mais claros.
    pt = _first_literal_lang(g, doc, RDFS.label, "pt")
    if pt: return pt
    en = _first_literal_lang(g, doc, RDFS.label, "en")
    if en: return en

    pt = _first_literal_lang(g, doc, DCTERMS.title, "pt")
    if pt: return pt
    en = _first_literal_lang(g, doc, DCTERMS.title, "en")
    if en: return en

    for lit in g.objects(doc, RDFS.label):
        return str(lit)
    for lit in g.objects(doc, DCTERMS.title):
        return str(lit)

    return str(doc)

def _frag_label(g: Graph, frag) -> str:
    for lit in g.objects(frag, RDFS.label):
        return str(lit)
    return str(frag)

def _frag_identifier(g: Graph, frag) -> Optional[int]:
    ident = next(iter(g.objects(frag, DCTERMS.identifier)), None)
    if ident is None:
        return None
    try:
        return int(str(ident))
    except Exception:
        return None

def _frag_to_doc(g: Graph, frag):
    for p in (ONTO.isFragmentOfDocument, DCTERMS.isPartOf):
        doc = next(iter(g.objects(frag, p)), None)
        if doc is not None:
            return doc

    doc = next(iter(g.subjects(ONTO.hasFragment, frag)), None)
    if doc is not None:
        return doc
    doc = next(iter(g.subjects(DCTERMS.hasPart, frag)), None)
    if doc is not None:
        return doc

    return None

def _doc_vector_node(g: Graph, doc):
    for p in (ONTO.hasProfileVector, ONTO.hasDocumentProfileVector):
        vec = next(iter(g.objects(doc, p)), None)
        if vec is not None:
            return vec
    return None

def _doc_vector_scores(g: Graph, vec) -> Tuple[Dict[str, float], Optional[str]]:
    scores: Dict[str, float] = {}
    # Aqui eu tento ler primeiro pelos predicados normalizados.
    for name, prop in VECTOR_ORDER:
        lit = next(iter(g.objects(vec, prop)), None)
        if lit is not None:
            try:
                scores[name] = float(lit)
            except Exception:
                pass

    mv = next(iter(g.objects(vec, ONTO.miVector)), None)
    mv_str = str(mv) if mv is not None else None

    if len(scores) < 8 and mv_str:
        parts = [p.strip() for p in mv_str.split(",")]
        if len(parts) == 8:
            vals: List[float] = []
            ok = True
            for p in parts:
                try:
                    v = float(p)
                    if v > 1.0:
                        v = v / 100.0
                    vals.append(v)
                except Exception:
                    ok = False
                    break
            if ok:
                for (name, _), v in zip(VECTOR_ORDER, vals):
                    scores.setdefault(name, v)

    return scores, mv_str

def _doc_vector_summary_lines(g: Graph, doc) -> Tuple[List[str], Optional[Tuple[str, float]]]:
    vec = _doc_vector_node(g, doc)
    if vec is None:
        return ["  (sem MIProfileVector agregado ligado ao documento via onto:hasProfileVector)"], None

    scores, mv = _doc_vector_scores(g, vec)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    lines: List[str] = []
    if mv:
        lines.append(f"  - miVector: {mv}")
    if ranking:
        top_name, top_val = ranking[0]
        lines.append(f"  - TOP (documento): {top_name} (score={top_val:.4f})")
        lines.append("  - Ranking (top 3):")
        for name, val in ranking[:3]:
            lines.append(f"      • {name}: {val:.4f}")
        return lines, (top_name, top_val)

    return ["  (vetor presente, mas sem scores interpretáveis)"], None


# ------------------------------------------------------------
# CQ1 — evocações por fragmento (activation vs element)
# ------------------------------------------------------------
CQ1_SPARQL = """
PREFIX onto: <https://techcoop.com.br/ontomi#>
PREFIX iao:  <http://purl.obolibrary.org/obo/IAO_>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?frag ?fragLabel ?intel ?intelLabel ?source ?score
WHERE {
  ?frag rdf:type onto:ExplanationFragment .
  OPTIONAL { ?frag rdfs:label ?fragLabel . }

  {
    ?frag onto:hasActivation ?act .
    ?act iao:0000136 ?intel .
    OPTIONAL { ?act onto:hasActivationScore ?score . }
    BIND("activation" AS ?source)
  }
  UNION
  {
    ?frag onto:usesElement ?el .
    ?el   onto:evokesIntelligence ?intel .
    BIND("element" AS ?source)
  }

  ?intel rdfs:subClassOf* onto:IntelligenceDisposition .
  OPTIONAL { ?intel rdfs:label ?intelLabel . }
}
ORDER BY ?frag ?source ?intelLabel ?intel
"""

def _cq1_text(g: Graph) -> Tuple[str, Dict[str, object]]:
    rows = list(g.query(CQ1_SPARQL))

    by_doc = {}
    for frag, fragLabel, intel, intelLabel, source, score in rows:
        doc = _frag_to_doc(g, frag)
        doc_id = str(doc) if doc is not None else "(sem_documento)"

        doc_entry = by_doc.setdefault(doc_id, {
            "doc_node": doc,
            "doc_label": _doc_label(g, doc) if doc is not None else "(sem_documento)",
            "frags": {}
        })

        fkey = (str(frag), str(fragLabel) if fragLabel else None)
        doc_entry["frags"].setdefault(fkey, []).append(
            (str(intel), str(intelLabel) if intelLabel else None, str(source), (float(score) if score else None))
        )

    out_lines = []

    for _, doc_info in sorted(by_doc.items(), key=lambda kv: kv[1]["doc_label"].lower()):
        out_lines.append(f"[Documento] {doc_info['doc_label']}")
        if doc_info["doc_node"] is not None:
            lines, _ = _doc_vector_summary_lines(g, doc_info["doc_node"])
            out_lines.extend(lines)
        out_lines.append("")

        # ordena fragmentos por dct:identifier quando existir
        frag_items = list(doc_info["frags"].items())
        def _frag_sort_key(item):
            (frag_iri, frag_lbl), _ = item
            ident = _frag_identifier(g, Namespace(frag_iri) if False else g.resource(frag_iri).identifier)  # safe no-op
            # rdflib resource trick acima pode não existir; então usa busca direta:
            ident2 = _frag_identifier(g, frag_iri) if isinstance(frag_iri, object) else None
            # fallback robusto: tenta pegar ident via Graph com URIRef
            try:
                from rdflib import URIRef
                ident3 = _frag_identifier(g, URIRef(frag_iri))
            except Exception:
                ident3 = None
            ident_final = ident3 if ident3 is not None else (ident2 if ident2 is not None else 10**9)
            return (ident_final, (frag_lbl or frag_iri))
        # como o sort acima tem g.resource gambiarra, aqui fiz diferente
        def _frag_sort_key_safe(item):
            (frag_iri, frag_lbl), _ = item
            try:
                from rdflib import URIRef
                ident = _frag_identifier(g, URIRef(frag_iri))
            except Exception:
                ident = None
            ident = ident if ident is not None else 10**9
            return (ident, (frag_lbl or frag_iri))
        frag_items.sort(key=_frag_sort_key_safe)

        for (frag, fragLabel), items in frag_items:
            header = fragLabel if fragLabel else frag
            out_lines.append(f"[Fragmento] {header}")
            for intel, intelLabel, source, score in items:
                name = intelLabel if intelLabel else intel.rsplit("#", 1)[-1]
                if source == "activation":
                    if score is not None:
                        out_lines.append(f"  - {name} (via {source}, score={score:.4f})")
                    else:
                        out_lines.append(f"  - {name} (via {source})")
                else:
                    out_lines.append(f"  - {name} (via {source})")
            out_lines.append("")

        out_lines.append("")

    metrics = {
        "rows": len(rows),
        "documents": len(by_doc),
        "fragments": sum(len(d["frags"]) for d in by_doc.values()),
    }
    return "\n".join(out_lines).rstrip() + "\n", metrics


# ------------------------------------------------------------
# CQ2 — elementos/ativações agrupados por inteligência (por fragmento)
# ------------------------------------------------------------
CQ2_SPARQL = """
PREFIX onto: <https://techcoop.com.br/ontomi#>
PREFIX iao:  <http://purl.obolibrary.org/obo/IAO_>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
  ?frag ?fragLabel
  ?intel
  ?intelLabel_pt ?intelLabel_en
  ?evidenceType ?el ?elLabel_pt ?elLabel_en
  ?act ?score ?atype
WHERE {
  ?frag rdf:type onto:ExplanationFragment .
  OPTIONAL { ?frag rdfs:label ?fragLabel . }

  {
    ?frag onto:usesElement ?el .
    ?el   onto:evokesIntelligence ?intel .
    BIND("element" AS ?evidenceType)
    OPTIONAL { ?el    rdfs:label ?elLabel_pt FILTER (lang(?elLabel_pt) = "pt") }
    OPTIONAL { ?el    rdfs:label ?elLabel_en FILTER (lang(?elLabel_en) = "en") }
  }
  UNION
  {
    ?frag onto:hasActivation ?act .
    ?act  iao:0000136 ?intel .
    BIND("activation" AS ?evidenceType)
    OPTIONAL { ?act onto:hasActivationScore ?score . }
    OPTIONAL { ?act onto:hasActivationType  ?atype . }
  }

  ?intel rdfs:subClassOf* onto:IntelligenceDisposition .
  OPTIONAL { ?intel rdfs:label ?intelLabel_pt FILTER (lang(?intelLabel_pt) = "pt") }
  OPTIONAL { ?intel rdfs:label ?intelLabel_en FILTER (lang(?intelLabel_en) = "en") }
}
ORDER BY ?frag ?intel ?evidenceType ?el ?act
"""

def _prefer_label(pt, en, fallback_iri: str) -> str:
    if pt: return str(pt)
    if en: return str(en)
    return fallback_iri.rsplit("#", 1)[-1] if "#" in fallback_iri else fallback_iri

def _class_of(g: Graph, node) -> str:
    for cls, name in [
        (ONTO.Keyword, "Keyword"),
        (ONTO.ContextObject, "ContextObject"),
        (ONTO.DiscursiveStrategy, "DiscursiveStrategy"),
    ]:
        if (node, RDF.type, cls) in g:
            return name
    return "ExplanationElement"

def _cq2_text(g: Graph) -> Tuple[str, Dict[str, object]]:
    rows = list(g.query(CQ2_SPARQL))

    # per-fragment
    data_frag = {}

    # per-document (agregado)
    data_doc = {}

    for (frag, fragLabel,
         intel, intelLabel_pt, intelLabel_en,
         evidenceType, el, elLabel_pt, elLabel_en,
         act, score, atype) in rows:

        frag_id  = str(frag)
        frag_lbl = str(fragLabel) if fragLabel else None

        doc = _frag_to_doc(g, frag)
        doc_id = str(doc) if doc is not None else "(sem_documento)"
        doc_lbl = _doc_label(g, doc) if doc is not None else "(sem_documento)"

        intel_id   = str(intel)
        intel_name = _prefer_label(intelLabel_pt, intelLabel_en, intel_id)

        # ---------- fragment ----------
        frag_entry = data_frag.setdefault(frag_id, {"label": frag_lbl, "doc_id": doc_id, "doc_label": doc_lbl, "intels": {}})
        intel_entry = frag_entry["intels"].setdefault(intel_id, {
            "name": intel_name, "elements": set(), "activations": []
        })

        # ---------- document ----------
        doc_entry = data_doc.setdefault(doc_id, {"label": doc_lbl, "doc_node": doc, "intels": {}})
        intel_doc_entry = doc_entry["intels"].setdefault(intel_id, {
            "name": intel_name, "elements": set(), "activations": []
        })

        if str(evidenceType) == "element" and el is not None:
            el_id = str(el)
            el_name = _prefer_label(elLabel_pt, elLabel_en, el_id)
            etype = _class_of(g, el)
            intel_entry["elements"].add((etype, el_name))
            intel_doc_entry["elements"].add((etype, el_name))

        if str(evidenceType) == "activation" and act is not None:
            s = float(score) if score is not None else None
            at = str(atype).rsplit("#", 1)[-1] if atype is not None else None
            intel_entry["activations"].append({"act": str(act), "score": s, "type": at})
            intel_doc_entry["activations"].append({"act": str(act), "score": s, "type": at})

    out_lines = []

    # imprime por documento primeiro (o TOTAL)
    for doc_id, doc_info in sorted(data_doc.items(), key=lambda kv: kv[1]["label"].lower()):
        out_lines.append(f"[Documento] {doc_info['label']}")
        if doc_info["doc_node"] is not None:
            lines, _ = _doc_vector_summary_lines(g, doc_info["doc_node"])
            out_lines.extend(lines)

        # inteligências no nível do doc
        for _, intel_info in sorted(doc_info["intels"].items(), key=lambda kv: kv[1]["name"].lower()):
            out_lines.append(f"  > Inteligência (doc): {intel_info['name']}")
            els = sorted(list(intel_info["elements"]), key=lambda e: (e[0], e[1].lower()))
            if els:
                out_lines.append("    - Elementos (agregado no doc):")
                for etype, ename in els:
                    out_lines.append(f"        • {etype}: {ename}")

            acts = intel_info["activations"]
            if acts:
                # pequeno resumo
                scores = [a["score"] for a in acts if a["score"] is not None]
                prim = sum(1 for a in acts if (a.get("type") or "").lower() == "primary")
                sec  = sum(1 for a in acts if (a.get("type") or "").lower() == "secondary")
                if scores:
                    out_lines.append(f"    - Ativações (doc): n={len(acts)}, primárias={prim}, secundárias={sec}, max={max(scores):.4f}")
                else:
                    out_lines.append(f"    - Ativações (doc): n={len(acts)}, primárias={prim}, secundárias={sec}")
        out_lines.append("")
        out_lines.append("")

    frags_by_doc = {}
    for frag_id, info in data_frag.items():
        frags_by_doc.setdefault(info["doc_id"], {"doc_label": info["doc_label"], "frags": []})
        frags_by_doc[info["doc_id"]]["frags"].append((frag_id, info))

    for doc_id, bucket in sorted(frags_by_doc.items(), key=lambda kv: kv[1]["doc_label"].lower()):
        bucket["frags"].sort(key=lambda t: (_frag_identifier(g, t[0]) if False else 10**9, (t[1]["label"] or t[0]).lower()))
        def _safe_ident(frag_id: str) -> int:
            try:
                from rdflib import URIRef
                ident = _frag_identifier(g, URIRef(frag_id))
                return ident if ident is not None else 10**9
            except Exception:
                return 10**9
        bucket["frags"].sort(key=lambda t: (_safe_ident(t[0]), (t[1]["label"] or t[0]).lower()))

        out_lines.append(f"[Documento — Fragmentos] {bucket['doc_label']}")
        out_lines.append("")

        for frag_id, frag_info in bucket["frags"]:
            header = frag_info["label"] if frag_info["label"] else frag_id
            out_lines.append(f"[Fragmento] {header}")
            for _, intel_info in sorted(frag_info["intels"].items(), key=lambda kv: kv[1]["name"].lower()):
                out_lines.append(f"  > Inteligência: {intel_info['name']}")
                els = sorted(list(intel_info["elements"]), key=lambda e: (e[0], e[1].lower()))
                if els:
                    out_lines.append("    - Elementos:")
                    for etype, ename in els:
                        out_lines.append(f"        • {etype}: {ename}")
                if intel_info["activations"]:
                    out_lines.append("    - Ativações:")
                    for a in intel_info["activations"]:
                        parts = []
                        if a["score"] is not None:
                            parts.append(f"score={a['score']:.4f}")
                        if a["type"]:
                            parts.append(f"tipo={a['type']}")
                        out_lines.append(f"        • " + (", ".join(parts) if parts else "(sem score/tipo)"))
            out_lines.append("")
        out_lines.append("")

    metrics = {
        "rows": len(rows),
        "documents": len(data_doc),
        "fragments": len(data_frag),
    }
    return "\n".join(out_lines).rstrip() + "\n", metrics


# ------------------------------------------------------------
# CQ3 — top inteligência por fragmento (max score)
# ------------------------------------------------------------
CQ3_SPARQL = """
PREFIX onto: <https://techcoop.com.br/ontomi#>
PREFIX iao:  <http://purl.obolibrary.org/obo/IAO_>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
  ?frag ?fragLabel
  ?intel ?intelLabel_pt ?intelLabel_en
  ?score ?atype
WHERE {
  ?frag rdf:type onto:ExplanationFragment .
  OPTIONAL { ?frag rdfs:label ?fragLabel . }

  ?frag onto:hasActivation ?act .
  ?act  iao:0000136 ?intel .
  ?act  onto:hasActivationScore ?score .

  OPTIONAL { ?act onto:hasActivationType ?atype . }

  ?intel rdfs:subClassOf* onto:IntelligenceDisposition .
  OPTIONAL { ?intel rdfs:label ?intelLabel_pt FILTER (lang(?intelLabel_pt) = "pt") }
  OPTIONAL { ?intel rdfs:label ?intelLabel_en FILTER (lang(?intelLabel_en) = "en") }
}
ORDER BY ?frag ?intel
"""

def _cq3_text(g: Graph) -> Tuple[str, Dict[str, object]]:
    rows = list(g.query(CQ3_SPARQL))

    by_frag = {}
    for frag, fragLabel, intel, lbl_pt, lbl_en, score, atype in rows:
        fid  = str(frag)
        flbl = str(fragLabel) if fragLabel else None
        iid  = str(intel)
        iname = _prefer_label(lbl_pt, lbl_en, iid)
        sc = float(score)
        typ = str(atype).rsplit("#", 1)[-1] if atype else None

        doc = _frag_to_doc(g, frag)
        doc_id = str(doc) if doc is not None else "(sem_documento)"
        doc_lbl = _doc_label(g, doc) if doc is not None else "(sem_documento)"

        frag_entry = by_frag.setdefault(fid, {"label": flbl, "doc_id": doc_id, "doc_label": doc_lbl, "acts": []})
        frag_entry["acts"].append({"intel_id": iid, "intel_name": iname, "score": sc, "type": typ})

    docs = {}
    for info in by_frag.values():
        docs.setdefault(info["doc_id"], info["doc_label"])

    out_lines = []

    for doc_id, doc_lbl in sorted(docs.items(), key=lambda kv: kv[1].lower()):
        out_lines.append(f"[Documento] {doc_lbl}")
        # tenta recuperar o nó do doc a partir de qualquer fragmento desse doc
        doc_node = None
        for fid, finfo in by_frag.items():
            if finfo["doc_id"] == doc_id:
                # pega o doc pelo primeiro frag
                try:
                    from rdflib import URIRef
                    doc_node = _frag_to_doc(g, URIRef(fid))
                except Exception:
                    doc_node = None
                break
        if doc_node is not None:
            lines, top = _doc_vector_summary_lines(g, doc_node)
            out_lines.extend(lines)
        else:
            out_lines.append("  (não foi possível identificar o nó do documento a partir dos fragmentos)")
        out_lines.append("")
        out_lines.append("")

    frags_by_doc = {}
    for fid, info in by_frag.items():
        frags_by_doc.setdefault(info["doc_id"], {"doc_label": info["doc_label"], "frags": []})
        frags_by_doc[info["doc_id"]]["frags"].append((fid, info))

    def _safe_ident(frag_id: str) -> int:
        try:
            from rdflib import URIRef
            ident = _frag_identifier(g, URIRef(frag_id))
            return ident if ident is not None else 10**9
        except Exception:
            return 10**9

    for doc_id, bucket in sorted(frags_by_doc.items(), key=lambda kv: kv[1]["doc_label"].lower()):
        out_lines.append(f"[Documento — Fragmentos] {bucket['doc_label']}")
        out_lines.append("")
        bucket["frags"].sort(key=lambda t: (_safe_ident(t[0]), (t[1]["label"] or t[0]).lower()))

        for fid, info in bucket["frags"]:
            header = info["label"] if info["label"] else fid
            acts = info["acts"]
            if not acts:
                out_lines.append(f"[Fragmento] {header}")
                out_lines.append("  (sem ativações)")
                out_lines.append("")
                continue

            max_score = max(a["score"] for a in acts)
            EPS = 1e-9
            tops = [a for a in acts if abs(a["score"] - max_score) <= EPS]

            primarias = [a for a in tops if (a["type"] or "").lower() == "primary"]
            if primarias:
                tops = primarias

            best_by_intel = {}
            for a in tops:
                key = (a["intel_id"], a["intel_name"])
                prev = best_by_intel.get(key)
                if (prev is None) or ((a["type"] or "").lower() == "primary" and (prev["type"] or "").lower() != "primary"):
                    best_by_intel[key] = a

            out_lines.append(f"[Fragmento] {header}")
            if len(best_by_intel) == 1:
                (_, iname), a = next(iter(best_by_intel.items()))
                t = f", tipo={a['type']}" if a["type"] else ""
                out_lines.append(f"  > TOP: {iname} (score={a['score']:.4f}{t})")
            else:
                out_lines.append("  > TOP (empate):")
                for (_, iname), a in sorted(best_by_intel.items(), key=lambda kv: kv[0][1].lower()):
                    t = f", tipo={a['type']}" if a["type"] else ""
                    out_lines.append(f"    - {iname} (score={a['score']:.4f}{t})")
            out_lines.append("")

        out_lines.append("")

    metrics = {
        "rows": len(rows),
        "documents": len(docs),
        "fragments": len(by_frag),
    }
    return "\n".join(out_lines).rstrip() + "\n", metrics

# ------------------------------------------------------------
# Runner para o batch (S6)
# ------------------------------------------------------------
def run_s6_from_files(
    *,
    in_ttl_path: str,
    out_dir: str,
) -> Dict[str, object]:
    """
    Executa CQ1/CQ2/CQ3 sobre o TTL (após S5) e salva outputs TXT no out_dir.
    Retorna métricas para o run_log.json.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    g = Graph()
    g.parse(in_ttl_path, format="turtle")

    cq1_txt, m1 = _cq1_text(g)
    cq2_txt, m2 = _cq2_text(g)
    cq3_txt, m3 = _cq3_text(g)

    f1 = outp / "cq1_evoked.txt"
    f2 = outp / "cq2_elements_by_intel.txt"
    f3 = outp / "cq3_top_intelligence.txt"

    f1.write_text(cq1_txt, encoding="utf-8")
    f2.write_text(cq2_txt, encoding="utf-8")
    f3.write_text(cq3_txt, encoding="utf-8")

    return {
        "status": "ok",
        "input": in_ttl_path,
        "outputs": {
            "cq1": f1.name,
            "cq2": f2.name,
            "cq3": f3.name,
        },
        "cq1": m1,
        "cq2": m2,
        "cq3": m3,
    }
