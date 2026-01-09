"""\
PT-BR: S2 — instancia fragmentos e documentos em RDF a partir da saída da S1.
EN: S2 — instantiates fragments/documents in RDF from the S1 output.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

BFO   = Namespace("http://purl.obolibrary.org/obo/BFO_")
IAO   = Namespace("http://purl.obolibrary.org/obo/IAO_")
ONTO  = Namespace("https://techcoop.com.br/ontomi#")
OWL   = Namespace("http://www.w3.org/2002/07/owl#")
RDFNS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFSNS= Namespace("http://www.w3.org/2000/01/rdf-schema#")
SH    = Namespace("http://www.w3.org/ns/shacl#")
XSD   = Namespace("http://www.w3.org/2001/XMLSchema#")
DCT   = Namespace("http://purl.org/dc/terms/")
VANN  = Namespace("http://purl.org/vocab/vann/")

FRAG_CLASS = ONTO.ExplanationFragment
DOC_CLASS  = ONTO.ExplanationDocument

def _bind_prefixes(g: Graph) -> None:
    # Eu mantenho os prefixes centralizados para manter o TTL legível.
    g.bind("bfo",  BFO)
    g.bind("iao",  IAO)
    g.bind("onto", ONTO)
    g.bind("owl",  OWL)
    g.bind("rdf",  RDFNS)
    g.bind("rdfs", RDFSNS)
    g.bind("sh",   SH)
    g.bind("xsd",  XSD)
    g.bind("dct",  DCT)
    g.bind("vann", VANN)

def instantiate_fragments_rdf(
    s1_output: Dict[str, Any],
    *,
    onto_path: str = "ontomi.ttl",
    base_instances_ns: Optional[str] = None,
    validate_ontology: bool = True,
    graph_mode: str = "full", 
) -> Tuple[Graph, List[URIRef]]:
    """
    S2 — Instancia ExplanationDocument + ExplanationFragment a partir do JSON da S1.

    graph_mode:
      - "full" (padrão): ontologia + instâncias no mesmo grafo (TTL grande, mas autocontido)
      - "instances": somente instâncias (TTL leve)
      - "instances+imports": instâncias + owl:imports (referencia a ontologia sem embed)
    """
    if graph_mode not in {"full", "instances", "instances+imports"}:
        raise ValueError("graph_mode inválido. Use: full | instances | instances+imports")

    if "doc_id" not in s1_output or "paragraphs" not in s1_output:
        raise ValueError("JSON de S1 inválido: campos obrigatórios 'doc_id' e 'paragraphs' ausentes.")

    doc_id: str = s1_output["doc_id"]
    paragraphs: List[Dict[str, Any]] = s1_output["paragraphs"]

    # Eu garanto que o namespace base sempre termina com # ou /.
    base_ns = base_instances_ns or (doc_id + "#")
    if not (base_ns.endswith("#") or base_ns.endswith("/")):
        base_ns = base_ns + "#"

    onto_file = Path(onto_path).resolve()
    if validate_ontology or graph_mode == "full":
        if not onto_file.exists():
            raise FileNotFoundError(f"Arquivo de ontologia não encontrado: {onto_file}")

    # 1) carrega ontologia somente se for necessário
    g_onto: Optional[Graph] = None
    if graph_mode == "full" or validate_ontology:
        g_onto = Graph()
        g_onto.parse(str(onto_file), format="turtle")

        # Checagem tolerante (avisa, mas não quebra)
        if validate_ontology and (FRAG_CLASS, None, None) not in g_onto:
            print(f"[WARN] {FRAG_CLASS} não encontrado na ontologia (checagem tolerante).")

    g = Graph()
    _bind_prefixes(g)

    if graph_mode == "full" and g_onto is not None:
        g += g_onto

    if graph_mode == "instances+imports":
        g.add((URIRef(doc_id), OWL.imports, URIRef(onto_file.as_uri())))

    doc_uri = URIRef(doc_id)
    g.add((doc_uri, RDF.type, DOC_CLASS))

    fragment_nodes: List[URIRef] = []

    for p in paragraphs:
        if "k" not in p or "text" not in p:
            continue

        k = int(p["k"])
        text = str(p["text"])
        lang = str(p.get("lang") or "und")
        is_ta = bool(p.get("is_title_or_abstract", False))

        frag_uri = URIRef(f"{base_ns}frag-{k:03d}")
        g.add((frag_uri, RDF.type, FRAG_CLASS))
        g.add((frag_uri, RDFS.label, Literal(text, lang=lang)))

        g.add((doc_uri, DCT.hasPart, frag_uri))
        g.add((frag_uri, DCT.isPartOf, doc_uri))

        g.add((frag_uri, DCT.identifier, Literal(k)))
        g.add((frag_uri, DCT.type, Literal("title_or_abstract" if is_ta else "body")))

        fragment_nodes.append(frag_uri)

    return g, fragment_nodes


def save_graph(g: Graph, path: str) -> None:
    g.serialize(destination=path, format="turtle")


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="S2 — Instanciação de ExplanationDocument/ExplanationFragment (OntoMI).")
    parser.add_argument("json_in", help="Caminho do JSON gerado pela S1.")
    parser.add_argument("--onto", default="ontomi.ttl", help="Caminho da ontologia OntoMI (TTL/OWL).")
    parser.add_argument("--out", default="instances_fragments.ttl", help="TTL de saída.")
    parser.add_argument("--base-ns", default=None, help="Namespace base (default: doc_id#).")
    parser.add_argument("--no-validate-ontology", action="store_true", help="Não faz parse/checagem da ontologia.")
    parser.add_argument(
        "--graph",
        default="full",
        choices=["full", "instances", "instances+imports"],
        help="Modo do grafo de saída: full (ontologia+instâncias), instances (só instâncias), instances+imports (instâncias + owl:imports).",
    )

    args = parser.parse_args()

    json_path = Path(args.json_in)
    with json_path.open("r", encoding="utf-8") as f:
        s1_output = json.load(f)

    g, frags = instantiate_fragments_rdf(
        s1_output,
        onto_path=args.onto,
        base_instances_ns=args.base_ns,
        validate_ontology=not args.no_validate_ontology,
        graph_mode=args.graph,
    )

    save_graph(g, args.out)
    print(f"Instâncias criadas: {len(frags)}")
    print(f"Grafo salvo em: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

