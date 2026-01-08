from __future__ import annotations

"""S1 — ingestão e segmentação de texto.

Este módulo **só** faz S1 (ingestão + limpeza + LID + segmentação em parágrafos)
usando a biblioteca `intelli3text`.

Design:
- API principal em forma de classe (`S1Ingestor`) para facilitar reuso no `run_batch`.
- Aceita **texto literal** ou **caminho de arquivo**.
- Tenta respeitar as opções do intelli3text, mas é tolerante a diferenças de
  assinatura (usa introspecção e ignora kwargs não suportadas).
- Se o intelli3text retornar **apenas 1 parágrafo** muito grande (caso comum em TXT
  com quebras irregulares), aplica um *fallback* de split (sentenças/chunks),
  para melhorar a granularidade do pipeline OntoMI.

Saída canônica (I/O para S2):
{
  "doc_id": "urn:doc:XYZ",
  "paragraphs": [
    {"k": 0, "text": "...", "lang": "pt", "is_title_or_abstract": true},
    {"k": 1, "text": "...", "lang": "pt", "is_title_or_abstract": false},
    ...
  ]
}
"""

import argparse
import json
import os
import re
import sys
import tempfile
import uuid
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------
# intelli3text
# ---------------------------------------------------------------------
from intelli3text import PipelineBuilder, Intelli3Config

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _is_url(s: str) -> bool:
    return bool(_URL_RE.match(s.strip()))


def _safe_intelli3_config(**kwargs) -> Intelli3Config:
    """Cria Intelli3Config filtrando kwargs não suportados pela versão instalada."""
    sig = inspect.signature(Intelli3Config)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return Intelli3Config(**filtered)  # type: ignore[arg-type]


def _materialize_source_for_pipeline(source: str, as_text: bool) -> Tuple[str, Optional[str]]:
    """Retorna (path_or_url, tmp_path) para uso no Pipeline.process(...)."""
    if as_text:
        tmp = tempfile.NamedTemporaryFile(prefix="i3_", suffix=".txt", delete=False)
        try:
            tmp.write(source.encode("utf-8"))
            tmp.flush()
        finally:
            tmp.close()
        return tmp.name, tmp.name

    # Se não for as_text:
    # - Se for URL, devolve a URL
    # - Se for arquivo existente, devolve o caminho
    # - Caso contrário, trate como literal e materialize em TXT
    if _is_url(source) or os.path.exists(source):
        return source, None

    tmp = tempfile.NamedTemporaryFile(prefix="i3_", suffix=".txt", delete=False)
    try:
        tmp.write(source.encode("utf-8"))
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name, tmp.name


def _looks_like_title(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if len(t) > 200:
        return False
    lowered = t.lower()
    if lowered.startswith(("resumo:", "abstract:", "sumário:", "summary:")):
        return False
    # pouco sinal de pontuação final
    if re.search(r"[.!?]\s*$", t):
        return False
    # baixa densidade de vírgulas/pontos
    punct = len(re.findall(r"[,;:]", t))
    return punct <= 2


def _mark_title_abstract(paragraphs: List[Dict[str, Any]], *, force_title: bool, title_scan_k: int, title_max_chars: int) -> None:
    """Marca `is_title_or_abstract` in-place, tentando garantir pelo menos um título."""
    if not paragraphs:
        return

    # zera
    for p in paragraphs:
        p["is_title_or_abstract"] = False

    # 1) detecta abstract/resumo explícito
    for p in paragraphs[: max(5, title_scan_k)]:
        tx = (p.get("text") or "").strip().lower()
        if tx.startswith(("resumo:", "abstract:", "sumário:", "summary:")):
            p["is_title_or_abstract"] = True

    # 2) escolhe candidato a título
    title_idx: Optional[int] = None
    scan = paragraphs[: max(1, title_scan_k)]
    for i, p in enumerate(scan):
        tx = (p.get("text") or "").strip()
        if not tx:
            continue
        if len(tx) <= title_max_chars and _looks_like_title(tx):
            title_idx = i
            break

    # fallback: primeiro parágrafo não-vazio dentro do limite
    if title_idx is None:
        for i, p in enumerate(scan):
            tx = (p.get("text") or "").strip()
            if tx and len(tx) <= title_max_chars:
                title_idx = i
                break

    # fallback final: força título no primeiro
    if title_idx is None and force_title:
        title_idx = 0

    if title_idx is not None and 0 <= title_idx < len(paragraphs):
        paragraphs[title_idx]["is_title_or_abstract"] = True


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long_text_to_chunks(text: str, *, max_chars: int, min_chars: int) -> List[str]:
    """Divide texto grande em chunks ~max_chars tentando respeitar fronteiras de sentença."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # tenta sentença
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sents) <= 1:
        # fallback: quebra dura
        out = []
        start = 0
        while start < len(text):
            out.append(text[start : start + max_chars].strip())
            start += max_chars
        return [o for o in out if o]

    out: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        chunk = " ".join(buf).strip()
        if chunk:
            out.append(chunk)
        buf = []
        buf_len = 0

    for s in sents:
        if not s:
            continue
        if buf_len + len(s) + 1 <= max_chars:
            buf.append(s)
            buf_len += len(s) + 1
        else:
            flush()
            buf.append(s)
            buf_len = len(s)

    flush()

    # garante min_chars: junta chunks pequenos com o próximo
    merged: List[str] = []
    i = 0
    while i < len(out):
        cur = out[i]
        if len(cur) < min_chars and i + 1 < len(out):
            cur = (cur + " " + out[i + 1]).strip()
            i += 2
            merged.append(cur)
        else:
            merged.append(cur)
            i += 1
    return merged


def _fallback_resegment(paragraphs: List[Dict[str, Any]], *, split_max_chars: int, split_min_chars: int) -> List[Dict[str, Any]]:
    """Se houver 1 parágrafo gigante, tenta resegmentar em vários."""
    if not paragraphs:
        return paragraphs

    # caso típico: 1 parágrafo só
    if len(paragraphs) == 1:
        tx = (paragraphs[0].get("text") or "").strip()
        chunks = _split_long_text_to_chunks(tx, max_chars=split_max_chars, min_chars=split_min_chars)
        if len(chunks) <= 1:
            return paragraphs
        base_lang = paragraphs[0].get("lang")
        return [{"k": i, "text": c, "lang": base_lang} for i, c in enumerate(chunks)]

    # se houver parágrafos muito longos, também chunkifica
    out: List[Dict[str, Any]] = []
    k = 0
    for p in paragraphs:
        tx = (p.get("text") or "").strip()
        if len(tx) > split_max_chars:
            chunks = _split_long_text_to_chunks(tx, max_chars=split_max_chars, min_chars=split_min_chars)
            for c in chunks:
                out.append({"k": k, "text": c, "lang": p.get("lang")})
                k += 1
        else:
            out.append({"k": k, "text": tx, "lang": p.get("lang")})
            k += 1
    return out


@dataclass
class S1Params:
    # intelli3text
    cleaners: List[str] = field(default_factory=lambda: ["ftfy", "clean_text", "pdf_breaks"])
    lid_primary: str = "fasttext"
    lid_fallback: Optional[str] = None
    languages_supported: Set[str] = field(default_factory=lambda: {"pt", "en", "es"})
    nlp_model_pref: str = "lg"  # lg|md|sm

    # controles de granularidade (intelli3text, se existir) + fallback local
    paragraph_min_chars: int = 30
    lid_min_chars: int = 60
    split_max_chars: int = 900
    split_min_chars: int = 120
    force_resegment_if_single: bool = True

    # marcação Título/Resumo
    force_title: bool = True
    title_scan_k: int = 3
    title_max_chars: int = 160


class S1Ingestor:
    def __init__(self, params: Optional[S1Params] = None):
        self.params = params or S1Params()

    def ingest(
        self,
        source: str,
        *,
        lang_hint: Optional[str] = None,
        doc_id: Optional[str] = None,
        as_text: bool = False,
    ) -> Dict[str, Any]:
        """Ingestão genérica: URL, caminho ou texto (as_text=True)."""
        path_or_url, tmp_path = _materialize_source_for_pipeline(source, as_text)

        # restringe idiomas se tiver hint
        languages_supported = set(self.params.languages_supported)
        if lang_hint in {"pt", "en", "es"}:
            languages_supported = {lang_hint}

        cfg = _safe_intelli3_config(
            cleaners=self.params.cleaners,
            lid_primary=self.params.lid_primary,
            lid_fallback=self.params.lid_fallback,
            languages_supported=languages_supported,
            nlp_model_pref=self.params.nlp_model_pref,
            paragraph_min_chars=self.params.paragraph_min_chars,
            lid_min_chars=self.params.lid_min_chars,
        )

        try:
            pipeline = PipelineBuilder(cfg).build()
            res = pipeline.process(path_or_url)

            # mapeia para estrutura do projeto
            paras: List[Dict[str, Any]] = []
            for i, p in enumerate(res.get("paragraphs", []) or []):
                tx = p.get("cleaned") or p.get("raw") or ""
                lang = p.get("language") or "und"
                paras.append({"k": i, "text": tx, "lang": lang})

            # fallback para quando tudo vira 1 parágrafo
            if self.params.force_resegment_if_single:
                paras = _fallback_resegment(
                    paras,
                    split_max_chars=self.params.split_max_chars,
                    split_min_chars=self.params.split_min_chars,
                )

            # garante doc_id
            if doc_id is None:
                doc_id = f"urn:doc:{uuid.uuid4()}"

            # marca título/resumo
            _mark_title_abstract(
                paras,
                force_title=self.params.force_title,
                title_scan_k=self.params.title_scan_k,
                title_max_chars=self.params.title_max_chars,
            )

            # reindexa k
            for i, p in enumerate(paras):
                p["k"] = i

            return {"doc_id": doc_id, "paragraphs": paras}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def ingest_text(self, text: str, *, lang_hint: Optional[str] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
        return self.ingest(text, lang_hint=lang_hint, doc_id=doc_id, as_text=True)

    def ingest_file(self, path: str | Path, *, lang_hint: Optional[str] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
        return self.ingest(str(path), lang_hint=lang_hint, doc_id=doc_id, as_text=False)

    def ingest_dir(
        self,
        source_dir: str | Path,
        out_dir: str | Path,
        *,
        lang_hint: Optional[str] = None,
        overwrite: bool = True,
    ) -> int:
        """Processa todos os .txt de uma pasta e salva output/<stem>/s1_output.json."""
        src = Path(source_dir)
        out = Path(out_dir)
        if not src.exists():
            raise FileNotFoundError(f"source_dir não existe: {src}")
        out.mkdir(parents=True, exist_ok=True)

        txts = sorted(src.glob("*.txt"))
        if not txts:
            return 0

        count = 0
        for txt in txts:
            stem = re.sub(r"[^\w\-. ]+", "", txt.stem).strip().replace(" ", "_")[:120]
            folder = out / stem
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "s1_output.json"
            if target.exists() and not overwrite:
                continue

            doc = self.ingest_file(txt, lang_hint=lang_hint, doc_id=f"urn:doc:{stem}")
            target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            count += 1
        return count


def _cli() -> int:
    p = argparse.ArgumentParser(description="S1 — Ingestão (intelli3text) -> s1_output.json")
    p.add_argument("source", nargs="?", default=None, help="URL, caminho de arquivo, ou texto literal (--as-text).")
    p.add_argument("--as-text", action="store_true", help="Interpreta source como TEXTO literal")

    # defaults alinhados ao run_batch
    p.add_argument("--source-dir", default="source", help="Processa uma pasta (quando source não é informado)")
    p.add_argument("--out-dir", default="output", help="Pasta de saída (modo pasta)")

    p.add_argument("--lang-hint", choices=["pt", "en", "es"], default=None, help="Dica de idioma")

    # intelli3text
    p.add_argument("--cleaners", default="ftfy,clean_text,pdf_breaks", help="Cleaners (CSV)")
    p.add_argument("--lid-primary", default="fasttext", help="LID primário")
    p.add_argument("--lid-fallback", default="none", help="LID fallback (ou 'none')")
    p.add_argument("--languages-supported", default="pt,en,es", help="Idiomas suportados (CSV)")
    p.add_argument("--nlp-size", default="lg", choices=["lg", "md", "sm"], help="Preferência spaCy")

    # granularidade
    p.add_argument("--paragraph-min-chars", type=int, default=30)
    p.add_argument("--lid-min-chars", type=int, default=60)
    p.add_argument("--split-max-chars", type=int, default=900)
    p.add_argument("--split-min-chars", type=int, default=120)
    p.add_argument("--no-resegment", action="store_true", help="Desliga fallback de resegmentação")

    # título
    p.add_argument("--no-force-title", action="store_true", help="Não força título (marca só se detectar)")
    p.add_argument("--title-scan-k", type=int, default=3)
    p.add_argument("--title-max-chars", type=int, default=160)

    p.add_argument("--json-out", default=None, help="(modo arquivo/texto) salva JSON neste caminho")

    args = p.parse_args()

    params = S1Params(
        cleaners=[c.strip() for c in args.cleaners.split(",") if c.strip()],
        lid_primary=args.lid_primary,
        lid_fallback=None if args.lid_fallback == "none" else args.lid_fallback,
        languages_supported=set(x.strip() for x in args.languages_supported.split(",") if x.strip()),
        nlp_model_pref=args.nlp_size,
        paragraph_min_chars=args.paragraph_min_chars,
        lid_min_chars=args.lid_min_chars,
        split_max_chars=args.split_max_chars,
        split_min_chars=args.split_min_chars,
        force_resegment_if_single=not args.no_resegment,
        force_title=not args.no_force_title,
        title_scan_k=args.title_scan_k,
        title_max_chars=args.title_max_chars,
    )

    ing = S1Ingestor(params)

    # MODO 1: source informado (arquivo/url/texto)
    if args.source is not None:
        doc = ing.ingest(args.source, as_text=args.as_text, lang_hint=args.lang_hint)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"doc_id: {doc['doc_id']} | paragraphs: {len(doc['paragraphs'])}")
        for p_ in doc["paragraphs"][:10]:
            flag = "T/A" if p_.get("is_title_or_abstract") else "corpo"
            prev = (p_["text"][:80] + "…") if len(p_["text"]) > 80 else p_["text"]
            print(f"[{p_['k']:03d}] ({p_['lang']}, {flag}) {prev}")
        return 0

    # MODO 2: pasta
    n = ing.ingest_dir(args.source_dir, args.out_dir, lang_hint=args.lang_hint)
    print(f"S1 concluído. Processados: {n} | out_dir: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
