# from __future__ import annotations

# import json
# from typing import Dict, Any, List, Tuple, Optional
# from pathlib import Path
# from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

# # ------------------------------------------------------------------------------------
# # Namespaces estritamente conforme a sua OntoMI
# # ------------------------------------------------------------------------------------
# BFO   = Namespace("http://purl.obolibrary.org/obo/BFO_")
# IAO   = Namespace("http://purl.obolibrary.org/obo/IAO_")
# ONTO  = Namespace("https://techcoop.com.br/ontomi#")
# OWL   = Namespace("http://www.w3.org/2002/07/owl#")
# RDFNS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
# RDFSNS= Namespace("http://www.w3.org/2000/01/rdf-schema#")
# SH    = Namespace("http://www.w3.org/ns/shacl#")
# XSD   = Namespace("http://www.w3.org/2001/XMLSchema#")
# DCT   = Namespace("http://purl.org/dc/terms/")
# VANN  = Namespace("http://purl.org/vocab/vann/")

# # ------------------------------------------------------------------------------------
# # Função principal da S2: cria o grafo de instâncias de ExplanationFragment
# # ------------------------------------------------------------------------------------
# def instantiate_fragments_rdf(
#     s1_output: Dict[str, Any],
#     *,
#     onto_path: str = "ontomi.ttl",
#     base_instances_ns: Optional[str] = None
# ) -> Tuple[Graph, List[URIRef]]:
#     """
#     S2 — Instanciação de fragments em RDF estritamente baseada na OntoMI.

#     s1_output:
#       {
#         "doc_id": "urn:doc:XYZ",
#         "paragraphs": [
#           { "k": 0, "text": "...", "lang": "pt", "is_title_or_abstract": true },
#           ...
#         ]
#       }
#     """
#     # --------------------------
#     # 1) Carregar ontologia
#     # --------------------------
#     onto_file = Path(onto_path)
#     if not onto_file.exists():
#         raise FileNotFoundError(f"Arquivo de ontologia não encontrado: {onto_file.resolve()}")

#     g = Graph()
#     g.parse(str(onto_file), format="turtle")

#     # Binds (legibilidade ao serializar)
#     g.bind("bfo",  BFO)
#     g.bind("iao",  IAO)
#     g.bind("onto", ONTO)
#     g.bind("owl",  OWL)
#     g.bind("rdf",  RDFNS)
#     g.bind("rdfs", RDFSNS)
#     g.bind("sh",   SH)
#     g.bind("xsd",  XSD)
#     g.bind("dct",  DCT)
#     g.bind("vann", VANN)

#     # --------------------------
#     # 2) Preparar base de URIs
#     # --------------------------
#     if "doc_id" not in s1_output or "paragraphs" not in s1_output:
#         raise ValueError("JSON de S1 inválido: campos obrigatórios 'doc_id' e 'paragraphs' ausentes.")

#     doc_id: str = s1_output["doc_id"]
#     paragraphs: List[Dict[str, Any]] = s1_output["paragraphs"]

#     base_ns = base_instances_ns or (doc_id + "#")

#     # --------------------------
#     # 3) Criar instâncias de Fragment
#     # --------------------------
#     fragment_nodes: List[URIRef] = []
#     FRAG_CLASS = ONTO.ExplanationFragment  # classe definida na OntoMI

#     for p in paragraphs:
#         if "k" not in p or "text" not in p:
#             # ignora entradas incompletas (p.ex., se vier algo fora do contrato)
#             continue

#         k = int(p["k"])
#         text = str(p["text"])
#         lang = str(p.get("lang") or "und")

#         frag_uri = URIRef(f"{base_ns}frag-{k:03d}")
#         g.add((frag_uri, RDF.type, FRAG_CLASS))

#         g.add((frag_uri, RDFS.label, Literal(text, lang=lang)))

#         fragment_nodes.append(frag_uri)

#     return g, fragment_nodes


# # ------------------------------------------------------------------------------------
# # Utilitário de salvamento
# # ------------------------------------------------------------------------------------
# def save_graph(g: Graph, path: str) -> None:
#     g.serialize(destination=path, format="turtle")


# # ------------------------------------------------------------------------------------
# # CLI: lê JSON da S1 e instancia
# #   python s2_instantiate.py s1_output.json --onto ontomi.ttl --out instances.ttl
# # ------------------------------------------------------------------------------------
# def _cli() -> int:
#     import argparse

#     parser = argparse.ArgumentParser(description="S2 — Instanciação de ExplanationFragment (OntoMI).")
#     parser.add_argument("json_in", help="Caminho do JSON gerado pela S1 (ingest_paragraphs).")
#     parser.add_argument("--onto", default="ontomi.ttl", help="Caminho da ontologia OntoMI (TTL/OWL).")
#     parser.add_argument("--out", default=None, help="Caminho do TTL de saída. Default: <json_in>.ttl")
#     parser.add_argument("--base-ns", default=None, help="Namespace base para instâncias (senão usa 'doc_id#').")
#     args = parser.parse_args()

#     json_path = Path(args.json_in)
#     if not json_path.exists():
#         raise FileNotFoundError(f"JSON de entrada não encontrado: {json_path.resolve()}")

#     with json_path.open("r", encoding="utf-8") as f:
#         s1_output = json.load(f)

#     g, frags = instantiate_fragments_rdf(
#         s1_output,
#         onto_path=args.onto,
#         base_instances_ns=args.base_ns
#     )

#     out_path = args.out or ("instances_fragments.ttl")
    
#     #     out_file = "instances_fragments.ttl"
#     save_graph(g, out_path)

#     print(f"Instâncias criadas: {len(frags)}")
#     print(f"Grafo salvo em: {out_path}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(_cli())
from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

# Namespaces conforme OntoMI
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
DOC_CLASS  = ONTO.ExplanationDocument  # se existir na sua OntoMI (provável)

def _bind_prefixes(g: Graph) -> None:
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
    graph_mode: str = "full",  # "full" | "instances" | "instances+imports"
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

    # 2) grafo de saída
    g = Graph()
    _bind_prefixes(g)

    # modo FULL: embute ontologia
    if graph_mode == "full" and g_onto is not None:
        g += g_onto

    # modo instances+imports: referencia ontologia sem embed (bem útil)
    if graph_mode == "instances+imports":
        # se você tiver um IRI “público” da OntoMI, use ele aqui.
        # Caso contrário, você pode importar a URL oficial do namespace (se existir),
        # ou deixar o imports como um file:// local (depende do seu uso).
        g.add((URIRef(doc_id), OWL.imports, URIRef(onto_file.as_uri())))

    # 3) documento
    doc_uri = URIRef(doc_id)
    g.add((doc_uri, RDF.type, DOC_CLASS))

    fragment_nodes: List[URIRef] = []

    # 4) fragments
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

        # ligações doc <-> frag
        g.add((doc_uri, DCT.hasPart, frag_uri))
        g.add((frag_uri, DCT.isPartOf, doc_uri))

        # metadados mínimos
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

