from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from rdflib import Graph
from pyshacl import validate
from rdflib.namespace import Namespace

SH = Namespace("http://www.w3.org/ns/shacl#")

def run_s7_shacl_validate_from_files(
    *,
    data_path: str,
    shacl_path: str,
    out_report_ttl_path: str,
    out_report_txt_path: Optional[str] = None,
    inference: str = "rdfs",
    advanced: bool = True,
    allow_infos: bool = True,
    allow_warnings: bool = True,
) -> Dict[str, object]:
    """
    Runner para o batch:
      - valida data_path contra shacl_path (usando também shacl_path como ont_graph)
      - salva report ttl e (opcionalmente) report txt
      - retorna métricas para run_log.json
    """
    data_g = Graph()
    data_g.parse(Path(data_path).as_posix(), format="turtle")

    shacl_g = Graph()
    shacl_g.parse(Path(shacl_path).as_posix(), format="turtle")

    ont_g = Graph()
    ont_g.parse(Path(shacl_path).as_posix(), format="turtle")

    conforms, report_graph, report_text = validate(
        data_graph=data_g,
        shacl_graph=shacl_g,
        ont_graph=ont_g,
        inference=inference,
        abort_on_first=False,
        allow_infos=allow_infos,
        allow_warnings=allow_warnings,
        advanced=advanced,
        meta_shacl=False,
        js=False,
        debug=False,
    )

    results_count = 0
    for _ in report_graph.objects(None, SH.result):
        results_count += 1

    out_report_ttl = Path(out_report_ttl_path)
    out_report_ttl.parent.mkdir(parents=True, exist_ok=True)
    report_graph.serialize(destination=out_report_ttl.as_posix(), format="turtle")

    if out_report_txt_path:
        Path(out_report_txt_path).write_text(report_text.strip() + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "conforms": bool(conforms),
        "results_count": int(results_count),
        "report_ttl": out_report_ttl.name,
        "report_txt": Path(out_report_txt_path).name if out_report_txt_path else None,
        "inference": inference,
        "advanced": bool(advanced),
        "allow_infos": bool(allow_infos),
        "allow_warnings": bool(allow_warnings),
    }

def main():
    DATA_PATH  = Path("instances_fragments_profile.ttl")
    SHACL_PATH = Path("ontomi.ttl")

    if not DATA_PATH.exists():
        print(f"[ERRO] Arquivo de dados não encontrado: {DATA_PATH.resolve()}")
        sys.exit(1)
    if not SHACL_PATH.exists():
        print(f"[ERRO] Ontologia (SHACL) não encontrada: {SHACL_PATH.resolve()}")
        sys.exit(1)

    data_g  = Graph()
    data_g.parse(DATA_PATH.as_posix(), format="turtle")

    shacl_g = Graph()
    shacl_g.parse(SHACL_PATH.as_posix(), format="turtle")

    ont_g = Graph()
    ont_g.parse(SHACL_PATH.as_posix(), format="turtle")

    conforms, report_graph, report_text = validate(
        data_graph=data_g,
        shacl_graph=shacl_g,
        ont_graph=ont_g,
        inference='rdfs',
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        advanced=True,
        meta_shacl=False,
        js=False,
        debug=False,
    )

    print("=" * 60)
    print("VALIDAÇÃO SHACL — OntoMI")
    print("=" * 60)
    print(f"Conforme? {'SIM' if conforms else 'NÃO'}")
    print("-" * 60)
    print(report_text.strip())
    print("-" * 60)

    out_ttl = Path("shacl_report.ttl")
    report_graph.serialize(destination=out_ttl.as_posix(), format="turtle")
    print(f"Relatório SHACL salvo em: {out_ttl.as_posix()}")

    sys.exit(0 if conforms else 2)

if __name__ == "__main__":
    main()
