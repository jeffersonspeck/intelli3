# README — Pipeline Intelli3 / OntoMI (S1 → S5)

Este repositório tem 5 scripts sequenciais:

* **S1 (`s1_ingest.py`)**: ingere texto (arquivo/URL/texto literal), limpa, segmenta em parágrafos e detecta idioma; gera **JSON**.
* **S2 (`s2_instantiate.py`)**: carrega a **OntoMI (TTL)** + instancia **`onto:ExplanationFragment`** para cada parágrafo do JSON; gera **TTL**.
* **S3 (`s3_evidences.py`)**: adiciona evidências (`Keyword`, `ContextObject`, `DiscursiveStrategy`) ligadas ao fragmento via **`onto:usesElement`** (opcionalmente **`onto:evokesIntelligence`**); gera **TTL**.
* **S4 (`s4_activations.py`)**: adiciona ativações (`onto:IntelligenceActivation`) por fragmento a partir de scores; define **primária** e **secundárias**; gera **TTL**.
* **S5 (`s5_profile_vector.py`)**: consolida ativações em um vetor **`onto:MIProfileVector`** (8D), normaliza e grava propriedades/percentuais; gera **TTL**.

---

## Dependências

* Python 3.x
* **rdflib**
* **intelli3text** (usado na S1)
* **evidences_api** (usado na S3 para montar payload/anotar via LLM; o contrato é consumido aqui, mas a lib não está nesses arquivos)

---

## Quickstart (fluxo recomendado)

### 1) S1 — gerar JSON a partir do texto

```bash
python s1_ingest.py ./meu_arquivo.pdf --lang-hint pt --json-out s1_output.json
```

### 2) S2 — instanciar fragments no grafo OntoMI (TTL)

```bash
python s2_instantiate.py s1_output.json --onto ontomi.ttl --out instances_fragments.ttl
```

### 3) S3 — adicionar evidências (gera `instances_fragments_evidences.ttl`)

> O `__main__` da S3 é um “CLI de teste mínimo” e hoje usa caminhos fixos (ex.: `instances_fragments.ttl` e `"arquivo.json"`). Ajuste esses paths no final do arquivo antes de rodar.

```bash
python s3_evidences.py
```

### 4) S4 — adicionar ativações (gera `instances_fragments_activations.ttl`)

> Também é “CLI de teste mínimo” com paths fixos.

```bash
python s4_activations.py
```

### 5) S5 — gerar vetores 8D por fragmento (gera `instances_fragments_profile.ttl`)

```bash
python s5_profile_vector.py
```

---

# S1 — `s1_ingest.py`

## O que faz

* Recebe **URL / caminho / texto literal**, materializa em arquivo temporário quando necessário e processa via **`intelli3text`** (limpeza, parágrafos e LID). 
* Usa config interna com `cleaners=["ftfy","clean_text","pdf_breaks"]`, `lid_primary="fasttext"` e idiomas suportados **pt/en/es** (ou restringe via dica). 
* Retorna JSON no contrato canônico para S2: `{ doc_id, paragraphs:[{k,text,lang,is_title_or_abstract}...] }`. 

## CLI (parâmetros)

```bash
python s1_ingest.py SOURCE [--as-text] [--lang-hint pt|en|es] [--json-out PATH]
```

* **SOURCE** (posicional): URL, caminho de arquivo, ou texto literal (com `--as-text`). 
* **--as-text**: interpreta `SOURCE` como **texto literal** e materializa em `.txt` para o pipeline. 
* **--lang-hint {pt,en,es}**: restringe o universo do LID ao idioma informado. 
* **--json-out PATH**: salva o JSON no caminho informado (além de imprimir um preview no console). 

## API (função)

```py
ingest_paragraphs(source, *, lang_hint=None, doc_id=None, as_text=False) -> dict
```

* **source**: URL/caminho/texto (se `as_text=True`). 
* **lang_hint**: `pt|en|es` (restringe `languages_supported`). 
* **doc_id**: se `None`, gera `urn:doc:<uuid>`. 
* **as_text**: força tratar `source` como literal. 

---

# S2 — `s2_instantiate.py`

## O que faz

* Carrega a ontologia (`ontomi.ttl`) no grafo, faz `bind` de namespaces para serialização legível. 
* Usa `doc_id + "#"`, a não ser que você passe **`base_instances_ns`**. 
* Para cada parágrafo com `k` e `text`, cria `doc#frag-000` etc, tipa como `onto:ExplanationFragment` e grava `rdfs:label` com o texto (lang). 

## CLI (parâmetros)

```bash
python s2_instantiate.py s1_output.json [--onto ontomi.ttl] [--out PATH] [--base-ns NS]
```

* **json_in** (posicional): caminho do JSON da S1. 
* **--onto**: caminho da OntoMI (TTL/OWL). 
* **--out**: TTL de saída (obs.: no código atual, se `None`, sai como `instances_fragments.ttl`). 
* **--base-ns**: namespace base para instâncias (senão usa `doc_id#`). 

## API (função)

```py
instantiate_fragments_rdf(s1_output, *, onto_path="ontomi.ttl", base_instances_ns=None) -> (Graph, [fragment_uris])
```

Parâmetros e contrato esperados estão documentados no próprio arquivo. 

---

# S3 — `s3_evidences.py`

## O que faz

Cria instâncias de evidências e liga cada uma ao fragmento via:

* `frag onto:usesElement evid` (sempre) 
* opcionalmente `evid onto:evokesIntelligence <ClasseDeInteligencia>` quando `intelligence` existe e `link_evokes_intelligence=True`. 

## API (função principal)

```py
add_evidences_rdf(g, fragment_nodes, evidences_payload, *, link_evokes_intelligence=True)
```

### Parâmetros

* **g**: grafo com OntoMI + fragments (S2). 
* **fragment_nodes**: lista de URIs dos fragments **na ordem** (índice `k` = posição). 
* **evidences_payload**: `dict[int, list[dict]]` no formato: 

  * `role`: `"Keyword" | "ContextObject" | "DiscursiveStrategy"` (qualquer outra coisa levanta erro). 
  * `text`: string obrigatória (itens incompletos são ignorados). 
  * `lang`: `pt|en|es` (default `"und"`). 
  * `intelligence` (opcional): label/localname; há tentativa de variantes simples (acentos e separadores). 
* **link_evokes_intelligence**: `bool` (default `True`). 

## “CLI” atual

O final do arquivo tem um **teste mínimo** (sem `argparse`) que:

* lê `instances_fragments.ttl`
* reconstrói `fragment_nodes` ordenando pelo IRI
* monta `evidences_payload` via `make_payload_from_s1_json_file("arquivo.json", annotate_fn=annotate_with_llm_default())`
* salva `instances_fragments_evidences.ttl` 

---

# S4 — `s4_activations.py`

## O que faz

Para cada fragmento `k`, recebe `{inteligencia -> score}` e cria instâncias `onto:IntelligenceActivation` ligadas ao fragmento, com score decimal e vínculo `iao:0000136` para a classe de inteligência. 

### Primária vs. secundárias (regra)

* **primária** = maior score do fragmento (em empate, fica a “primeira” no loop) e ganha:

  * `frag onto:hasPrimaryActivation act`
  * `act onto:hasActivationType onto:Primary` 
* **secundária** = score `>= theta * max_score`:

  * `frag onto:hasSecondaryActivation act`
  * `act onto:hasActivationType onto:Secondary` 
* **abaixo do limiar**: mantém em `onto:hasActivation`, mas ainda tipa `hasActivationType Secondary` “para satisfazer SHACL”. 

## API (função principal)

```py
add_activations_rdf(g, fragment_nodes, scores_by_fragment, *, theta=0.75, link_isAboutFragment=True)
```

* **scores_by_fragment**: `dict[int, dict[str,float]]` (k -> {nomeInteligencia -> score}). 
* **theta**: limiar relativo (default `0.75`). 
* **link_isAboutFragment**: se `True`, adiciona `act onto:isAboutFragment frag`. 
* Chaves de inteligência aceitam **label PT/EN** ou **localname** (o script tenta normalizar separadores; se não resolver, ignora silenciosamente). 

## “CLI” atual

Teste mínimo (paths fixos), salvando `instances_fragments_activations.ttl`. 

---

# S5 — `s5_profile_vector.py`

## O que faz

Para cada `ExplanationFragment`, soma scores das ativações por inteligência, monta vetor 8D em eixo fixo, **normaliza L1** e cria uma instância `onto:MIProfileVector` com propriedades:

1. `hasLinguisticScore`
2. `hasLogicalMathematicalScore`
3. `hasSpatialScore`
4. `hasBodilyKinestheticScore`
5. `hasMusicalScore`
6. `hasInterpersonalScore`
7. `hasIntrapersonalScore`
8. `hasNaturalistScore`

Além disso, opcionalmente grava uma string `"xx.xx,yy.yy,..."` em `onto:miVector`, e liga via `frag onto:hasProfileVector vec`. 

## API (função)

```py
build_profile_vectors(g, *, vec_places=4, write_percent_string=True)
```

* **vec_places**: casas decimais nas propriedades (default `4`). 
* **write_percent_string**: grava `onto:miVector` como percentuais (default `True`). 

## “CLI” atual

Lê `instances_fragments_activations.ttl` e salva `instances_fragments_profile.ttl`. 

---

## Observações práticas (para evitar “pegadinhas”)

* **Ordem dos fragments importa** em S3/S4: a função usa `enumerate(fragment_nodes)` e espera que o índice `k` do payload bata com a posição. No “teste mínimo” da S3 e S4, os fragments são reconstituídos e ordenados pelo IRI. 
* Em **S2**, o `--out` diz “Default: `<json_in>.ttl`”, mas o código atualmente usa `instances_fragments.ttl` quando `--out` não é passado. 
* Em **S3**, `role` é estrito (só 3 valores) e normaliza `_` e `-` antes de validar. 

Se você quiser, eu já te devolvo uma versão “de verdade” da S3 e S4 com **argparse completo** (aceitando `--in`, `--json`, `--out`, `--theta`, `--no-llm`, etc.) mantendo o comportamento atual como default.



####

Aqui vai um **script “batch”** que percorre `source/*.txt` e roda **S1 → S2 → S3 → (gera scores) → S4 → S5** para **cada arquivo**, salvando tudo em `out/<nome_do_arquivo>/`.

Crie um arquivo chamado **`run_batch.py`** na mesma pasta onde estão `s1_ingest.py`, `s2_instantiate.py`, `s3_evidences.py`, `s4_activations.py`, `s5_profile_vector.py` e `evidences_api.py`.

> Pré-requisitos de pastas:
>
> * `source/` (coloque aqui seus `.txt`)
> * `ontomi.ttl` no diretório atual (ou passe via `--onto`)

---

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

from rdflib import Graph, Namespace, URIRef, RDF

# Importa suas etapas
import s1_ingest
import s2_instantiate
import s3_evidences
import s4_activations
import s5_profile_vector

# Evidences API (LLM + YAKE + heurísticas)
from evidences_api import annotate_with_llm_default, _ollama_up, OLLAMA_HOST

ONTO = Namespace("https://techcoop.com.br/ontomi#")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_ttl(g: Graph, path: Path) -> None:
    g.serialize(destination=str(path), format="turtle")


def compute_scores_from_evidences(g: Graph, fragment_nodes: List[URIRef]) -> Dict[int, Dict[str, float]]:
    """
    Gera scores_by_fragment[k] = { "LocalNameDaClasseInteligencia": score_float }
    a partir das evidências que possuem onto:evokesIntelligence.

    Estratégia: contagem por inteligência (frequência).
    """
    scores_by_fragment: Dict[int, Dict[str, float]] = {}

    for k, frag in enumerate(fragment_nodes):
        counts: Dict[str, float] = {}

        # frag --onto:usesElement--> evid
        for _, _, evid in g.triples((frag, ONTO.usesElement, None)):
            # evid --onto:evokesIntelligence--> IntelligenceDispositionClass
            for _, _, intel_cls in g.triples((evid, ONTO.evokesIntelligence, None)):
                local = str(intel_cls).rsplit("#", 1)[-1]  # localname
                counts[local] = counts.get(local, 0.0) + 1.0

        if counts:
            scores_by_fragment[k] = counts

    return scores_by_fragment


def run_one_txt(
    txt_path: Path,
    *,
    onto_path: Path,
    out_root: Path,
    lang_hint: str | None,
    offline: bool,
    theta: float,
) -> None:
    stem = txt_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # S1: ingest -> JSON
    # -----------------------
    s1_json_path = out_dir / "s1_output.json"

    s1_out = s1_ingest.ingest_paragraphs(str(txt_path), lang_hint=lang_hint, as_text=False)

    # padroniza doc_id p/ ficar estável por arquivo (útil p/ URIs)
    s1_out["doc_id"] = f"urn:doc:{stem}"
    write_json(s1_json_path, s1_out)

    # -----------------------
    # S2: instanciar fragments -> TTL
    # -----------------------
    s2_ttl_path = out_dir / "instances_fragments.ttl"

    g2, fragment_nodes = s2_instantiate.instantiate_fragments_rdf(
        s1_out,
        onto_path=str(onto_path),
        base_instances_ns=f"urn:doc:{stem}#",
    )
    save_ttl(g2, s2_ttl_path)

    # -----------------------
    # S3: evidências (LLM/heurística) -> TTL
    # -----------------------
    s3_ttl_path = out_dir / "instances_fragments_evidences.ttl"

    if offline:
        annotate_fn = annotate_with_llm_default(use_llm=False)
    else:
        # se não estiver UP, cai automaticamente em offline (evita crash)
        if not _ollama_up(OLLAMA_HOST):
            print(f"[WARN] Ollama não respondeu em {OLLAMA_HOST}. Rodando S3 em modo offline para {stem}.", file=sys.stderr)
            annotate_fn = annotate_with_llm_default(use_llm=False)
        else:
            annotate_fn = annotate_with_llm_default()

    evidences_payload = annotate_fn(s1_out)  # Dict[int, List[Dict]]

    g3, _created = s3_evidences.add_evidences_rdf(
        g2,
        fragment_nodes,
        evidences_payload,
        link_evokes_intelligence=True,
    )
    save_ttl(g3, s3_ttl_path)

    # -----------------------
    # (extra) scores por fragment a partir de evokesIntelligence
    # -----------------------
    scores_by_fragment = compute_scores_from_evidences(g3, fragment_nodes)

    # -----------------------
    # S4: ativações -> TTL
    # -----------------------
    s4_ttl_path = out_dir / "instances_fragments_activations.ttl"

    g4, _acts = s4_activations.add_activations_rdf(
        g3,
        fragment_nodes,
        scores_by_fragment,
        theta=theta,
    )
    save_ttl(g4, s4_ttl_path)

    # -----------------------
    # S5: profile vector -> TTL
    # -----------------------
    s5_ttl_path = out_dir / "instances_fragments_profile.ttl"

    g5, _mapping = s5_profile_vector.build_profile_vectors(
        g4,
        vec_places=4,
        write_percent_string=True,
    )
    save_ttl(g5, s5_ttl_path)

    # Opcional: TTL final “principal”
    final_ttl = out_dir / f"{stem}.ttl"
    save_ttl(g5, final_ttl)

    print(f"[OK] {stem}")
    print(f"  - {final_ttl}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch runner: S1→S2→S3→S4→S5 para todos os .txt em /source")
    ap.add_argument("--source-dir", default="source", help="Pasta com .txt de entrada")
    ap.add_argument("--out-dir", default="out", help="Pasta de saída")
    ap.add_argument("--onto", default="ontomi.ttl", help="Caminho para ontomi.ttl")
    ap.add_argument("--lang-hint", choices=["pt", "en", "es"], default=None, help="Hint de idioma p/ S1")
    ap.add_argument("--offline", action="store_true", help="Força S3 sem LLM (heurística apenas)")
    ap.add_argument("--theta", type=float, default=0.75, help="Threshold de secundárias na S4 (theta * max)")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    onto_path = Path(args.onto)

    if not source_dir.exists():
        print(f"[ERRO] Pasta source não encontrada: {source_dir.resolve()}", file=sys.stderr)
        return 2
    if not onto_path.exists():
        print(f"[ERRO] Ontologia não encontrada: {onto_path.resolve()}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    txts = sorted(source_dir.glob("*.txt"))
    if not txts:
        print(f"[ERRO] Nenhum .txt encontrado em: {source_dir.resolve()}", file=sys.stderr)
        return 2

    for txt in txts:
        try:
            run_one_txt(
                txt,
                onto_path=onto_path,
                out_root=out_dir,
                lang_hint=args.lang_hint,
                offline=args.offline,
                theta=args.theta,
            )
        except Exception as e:
            print(f"[FAIL] {txt.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Como usar

1. Coloque seus `.txt` em:

```
source/
  texto1.txt
  texto2.txt
```

2. Rode:

```bash
python3 run_batch.py
```

3. Saída fica assim:

```
out/
  texto1/
    s1_output.json
    instances_fragments.ttl
    instances_fragments_evidences.ttl
    instances_fragments_activations.ttl
    instances_fragments_profile.ttl
    texto1.ttl
  texto2/
    ...
```

Se quiser rodar **sem LLM** (só heurística):

```bash
python3 run_batch.py --offline
```

Se quiser, eu também adapto esse runner pra aceitar `.pdf` e `.md` (mantendo “um arquivo → uma ontologia”).

#####

Fechado — eu já deixei tudo isso automatizado num **orquestrador em lote** que:

* roda **S1→S5** por **cada `.txt`**,
* executa também **CQ1/CQ2/CQ3 (S6)** (todos lendo `instances_fragments_profile.ttl`)   
* executa **SHACL (S7)** lendo `instances_fragments_profile.ttl` + `ontomi.ttl` e salvando `shacl_report.ttl`  
* mede **tempo por etapa e total por texto**,
* imprime no console: **TOP inteligência + miVector + tempo total**,
* salva um **`report.txt`** completo dentro da pasta de saída daquele texto (com vetores, TOP, outputs das CQs e SHACL, e tempos).

Ele segue o contrato do pipeline (S2 espera `doc_id` + `paragraphs` vindos da S1) , aplica o `theta` para **Primary/Secondary** na S4  e gera o **MIProfileVector** com as 8 dimensões na S5 .

[Download do run_batch.py](sandbox:/mnt/data/run_batch.py)

### Como usar

1. Coloque o `run_batch.py` **na mesma pasta** onde estão `s1_ingest.py ... s7_shacl_validate.py` (e seus módulos auxiliares).
2. Crie uma pasta `source/` e coloque lá seus `.txt`.
3. Garanta que `ontomi.ttl` esteja na pasta raiz (ou passe `--onto`).
4. Rode:

```bash
python run_batch.py --source-dir source --out-dir output --theta 0.75
```

Opcional (rodar S3 sem LLM):

```bash
python run_batch.py --no-llm
```

### Saídas

Para cada `source/ARQUIVO.txt`, ele cria `output/ARQUIVO/` contendo:

* `instances_fragments*.ttl` (S2..S5),
* `shacl_report.ttl` (S7),
* **`report.txt`** (com inteligência TOP, miVector, logs das CQs/SHACL e tempos por etapa e total).


#####

Pronto — ajustei seu batch para usar o **novo S5** (vetor do **documento** agregado + opção de vetor por **fragmento**), mantendo o restante do fluxo S1..S7 igual e imprimindo no console o **TOP + miVector** com base no **vetor do documento**.

[Baixar o run_batch ajustado](sandbox:/mnt/data/run_batch_updated.py)

### O que mudou (objetivamente)

* **S5 agora chama**:

  * `build_document_profile_vector(...)` (vetor agregado do documento, com `alpha_title_abstract` / `alpha_body` e `norm`)
  * e **opcionalmente** `build_fragment_profile_vectors(...)` (auditoria/debug), controlado por `--no-frag-vectors`
* O batch **não quebra** se você tiver o S5 novo em `s5_profile_vector.py` **ou** em `s5_profile_vector_v2.py` (ele tenta ambos).
* O batch também cria o campo `s1_out["fragments"]` a partir de `s1_out["paragraphs"]` (porque seu S1 usa `k`, e o S5 novo espera `idx`).

### Como usar

```bash
python run_batch_updated.py --source-dir source --out-dir output --onto ontomi.ttl
```

Com configurações do S5:

```bash
python run_batch_updated.py --norm l1 --alpha-title-abstract 1.3 --alpha-body 1.0
```

Para gerar **apenas** o vetor do documento (sem vetores por fragmento):

```bash
python run_batch_updated.py --no-frag-vectors
```

Se quiser, eu também já deixo esse arquivo como `run_batch.py` (mesmo conteúdo) e adapto o trecho do seu repositório exatamente no lugar certo (imports + bloco S5), mas com esse arquivo você já consegue substituir direto.
