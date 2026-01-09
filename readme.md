# Intelli3

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ontology](https://img.shields.io/badge/Ontology-RDF%2FOWL-informational)
![SHACL](https://img.shields.io/badge/Validation-SHACL-informational)
![LLM](https://img.shields.io/badge/LLM-Ollama%20Chat-informational)

Intelli3 is a **Python-based research prototype** that analyzes educational resources (primarily **text**) and produces an **explainable “Multiple Intelligences” profile** by combining:

- **Ontology-driven modeling** (OntoMI in RDF/TTL)
- **LLM-based annotation** (via Ollama chat, optional)
- **Evidence extraction** (e.g., YAKE candidates + contextual classification)
- **Validation** (SHACL constraints)

This repository is part of my **M.Sc. research in Computer Science**, supervised by **Prof. Sidgley Camargo de Andrade** and **Prof. Clodis Boscarioli**.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Documentation](#documentation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Key Parameters](#key-parameters)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Intelli3 Classifier](#intelli3-classifier)

## Repository Structure

| Path | Description |
| --- | --- |
| `docs/` | Technical documentation for each pipeline stage. |
| `source/` | Input materials to be analyzed (usually `.txt`). |
| `output/` | Pipeline outputs per document and batch reports. |
| `run_batch.py` | Batch orchestrator for stages S1–S7. |
| `s1_ingest.py` ... `s7_shacl_validate.py` | Stage scripts executed by the batch runner. |
| `evidences_api.py` | Evidence extraction and LLM annotation layer used by S3. |
| `ontomi.ttl` | OntoMI ontology (RDF/OWL). |

## Pipeline Overview

The pipeline is organized in sequential scripts, executed by `run_batch.py`:

1. **S1 — Ingest**: prepares raw inputs and segmentation (`s1_ingest.py`).
2. **S2 — RDF instantiation**: creates RDF instances (`s2_instantiate.py`).
3. **S3 — Evidences**: extracts evidences and MI signals (`s3_evidences.py`).
4. **S4 — Activations**: converts evidences to activation scores (`s4_activations.py`).
5. **S5 — Profile vector**: computes MI vectors (`s5_profile_vector.py`).
6. **S6 — Tests**: SPARQL-based competency questions (`s6_tests.py`).
7. **S7 — SHACL validation**: validates RDF (`s7_shacl_validate.py`).

### Dependency rule

If a stage is disabled, all subsequent stages are skipped.

## Documentation

Start with **[`docs/README.md`](docs/README.md)** for the stage index, then open the stage-specific files:

- [`docs/s1_ingest.md`](docs/s1_ingest.md)
- [`docs/s2_instantiate.md`](docs/s2_instantiate.md)
- [`docs/s3_evidences.md`](docs/s3_evidences.md)
- [`docs/s4_activations.md`](docs/s4_activations.md)
- [`docs/s5_profile_vector.md`](docs/s5_profile_vector.md)
- [`docs/s6_tests.md`](docs/s6_tests.md)
- [`docs/s7_shacl_validate.md`](docs/s7_shacl_validate.md)

## Installation

### 1) Python

Install **Python 3.10+** and create a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

### 2) Core dependencies

RDF/SHACL stack (S2+):

```bash
pip install rdflib pyshacl
```

Optional text processing dependencies (S1/S3):

```bash
pip install spacy unidecode yake ftfy clean-text requests
```

> **Note:** S1 uses your local module **`intelli3text`**. Install it as your environment requires:
>
> ```bash
> pip install -e path/to/intelli3text
> ```

### 3) spaCy models (if S1 is enabled)

```bash
python -m spacy download en_core_web_md
python -m spacy download pt_core_news_md
python -m spacy download es_core_news_md
```

### 4) LLM runtime (if S3 uses an LLM)

If `evidences_api.py` uses Ollama:

1. Install Ollama (OS-specific).
2. Ensure it is running (default: `http://127.0.0.1:11434`).
3. Pull your target model, e.g.:

```bash
ollama pull mistral:7b-instruct
```

## Quick Start

1) Put `.txt` files in `source/`.

```bash
mkdir -p source
# add .txt files to ./source/
```

2) Ensure the ontology is available at `ontomi.ttl` (or pass `--onto <path>`).

3) Run the pipeline:

```bash
python run_batch.py
```

## CLI Usage

### Basic

```bash
python run_batch.py --source-dir source --out-dir output
```

### Turn steps on/off

- `--no-s2` : disable S2 (and therefore S3–S7)
- `--no-s3` : disable S3 (and therefore S4–S7)
- `--no-s4` : disable S4 (and therefore S5–S7)
- `--no-s5` : disable S5 (and therefore S6–S7)
- `--no-s6` : disable S6 (and therefore S7)
- `--no-s7` : disable S7

Examples:

```bash
# Run only S1 (ingest JSON)
python run_batch.py --no-s2

# Run S1 + S2 only
python run_batch.py --no-s3

# Run full pipeline, but disable SHACL
python run_batch.py --no-s7
```

## Key Parameters

Below is a grouped reference for the most relevant parameters supported by the batch runner.
Defaults are the intended values documented by the project (the code is the source of truth).

### Paths

- `--source-dir` (default: `source`): folder containing `.txt` inputs.
- `--out-dir` (default: `output`): output root; a subfolder is created per document.

### S1 — Ingest (core)

- `--lang-hint` (default: `None`, choices: `pt|en|es`)
- `--cleaners` (default: `ftfy,clean_text,pdf_breaks`)
- `--lid-primary` (default: `fasttext`)
- `--lid-fallback` (default: `none`)
- `--languages` (default: `pt,en,es`)
- `--nlp-size` (default: `lg`, choices: `lg|md|sm`)

### S1 — Segmentation and fallback splitting

- `--paragraph-min-chars` (default: `30`)
- `--lid-min-chars` (default: `60`)
- `--split-max-chars` (default: `900`)
- `--split-min-chars` (default: `120`)
- `--no-resegment`

### S1 — Title/Abstract detection

- `--force-title`
- `--title-scan-k` (default: `3`)
- `--title-max-chars` (default: `160`)

### S2 — RDF instantiation

- `--onto` (default: `ontomi.ttl`)
- `--base-ns` (default: `None`)
- `--s2-graph` (default: `full`, `instances`, `instances+imports`)

### S3 — Evidences

- `--s3-no-llm`
- `--s3-no-evokes`

### S4 — Activations

- `--theta` (default: `0.75`)
- `--w-keyword` (default: `1.00`)
- `--w-context` (default: `1.25`)
- `--w-strategy` (default: `1.10`)

### S5 — MI profile vector

- `--s5-scope` (default: `both`)
- `--s5-norm` (default: `l1`, `l2`, `softmax`)
- `--s5-tau` (default: `1.0`)
- `--alpha-title` (default: `1.30`)
- `--alpha-body` (default: `1.00`)
- `--s5-vec-places` (default: `4`)
- `--s5-no-mi-vector-string`

### S7 — SHACL validation

- `--s7-inference` (default: `rdfs`, options: `none|rdfs|owlrl`)

## Outputs

Each input document generates `output/<doc_slug>/` with artifacts such as:

- `s1_output.json`
- `instances_fragments.ttl`
- `instances_fragments_evidences.ttl`
- `evidences_payload.json`
- `scores_by_fragment.json`
- `instances_fragments_activations.ttl`
- `instances_fragments_profile.ttl`
- `cq1_evoked.txt`, `cq2_elements_by_intel.txt`, `cq3_top_intelligence.txt`
- `shacl_report.ttl`, `shacl_report.txt`
- `run_log.json`

Batch-level output:

- `output/batch_report.json`

## Troubleshooting

### “ontologia não encontrada”

If S2 is enabled, `--onto` must exist (default: `ontomi.ttl`).

### S3 fails with LLM / Ollama errors

If your `evidences_api.py` uses an LLM:

- Ensure the LLM runtime is up.
- Ensure the target model is pulled and available.
- Or run with `--s3-no-llm` (only if your evidence logic supports it).

### SHACL doesn’t conform

Inspect `shacl_report.txt` and `shacl_report.ttl` in the document folder.

## Intelli3 Classifier

The classifier is a **decoupled module** that operates on the MI vectors generated by S1–S5.
It ranks documents by similarity to **reference cognitive profiles** and supports PoC evaluation.

See **[`classifier/README.md`](classifier/README.md)** for usage and experiments.
