# OntoMI Batch

This repository contains a **batch orchestrator** (`run_batch.py`) that runs a multi-step pipeline over **plain text files** (`.txt`) and produces:

- **JSON** outputs for ingestion + telemetry
- **RDF/Turtle (`.ttl`) graphs** with instantiated fragments, evidences, activations, and MI profile vectors
- **CQ test outputs** (CQ1/CQ2/CQ3)
- **SHACL validation reports**

The pipeline is organized in steps **S1 → S7** and is designed to be extended incrementally.

---

## What the batch does (default behavior)

When executed with no extra parameters, `run_batch.py`:

1. Reads all `.txt` files from `./source/`
2. For each file, creates `./output/<slug_of_filename>/`
3. Runs the pipeline steps (depending on which ones are enabled)
4. Writes step outputs inside each document folder and a `run_log.json`
5. Writes a `batch_report.json` at the root output folder

> **Important:** steps are chained with dependencies:
> - If **S2** is disabled, **S3..S7** won’t run.
> - If **S3** is disabled, **S4..S7** won’t run.
> - If **S4** is disabled, **S5..S7** won’t run.
> - If **S5** is disabled, **S6..S7** won’t run.
> - If **S6** is disabled, **S7** won’t run.

---

## Project layout

Recommended structure:

project_root/
run_batch.py
ontomi.ttl                # OntoMI ontology (required for S2+)
evidences_api.py          # Required by S3 (your implementation)
s1_ingest.py
s2_instantiate.py
s3_evidences.py
s4_activations.py
s5_profile_vector.py
s6_tests.py
s7_shacl_validate.py

source/
doc1.txt
doc2.txt

output/
<doc_slug>/
s1_output.json
instances_fragments.ttl
instances_fragments_evidences.ttl
evidences_payload.json
scores_by_fragment.json
instances_fragments_activations.ttl
instances_fragments_profile.ttl
cq1_evoked.txt
cq2_elements_by_intel.txt
cq3_top_intelligence.txt
shacl_report.ttl
shacl_report.txt
run_log.json
batch_report.json

---

## Requirements

### Python
- Python 3.10+ recommended (type hints, pathlib, dataclasses patterns)

### Key libraries (high-level)
This project typically depends on:

- `rdflib` (RDF graph manipulation, Turtle serialization)
- `pyshacl` (SHACL validation)
- `spacy` + language model(s) for the ingest step
- Your local/project dependencies:
  - `intelli3text` (used by S1)
  - `evidences_api.py` (required by S3)
  - If S3 uses an LLM, your runtime must also provide the needed LLM stack (e.g., Ollama, model available)

> Because S1 and S3 rely on **your local modules** (`intelli3text`, `evidences_api.py`), installation may be project-specific.

---

## Quick start

1) Create the input folder and place `.txt` files:

```bash
mkdir -p source
# put doc1.txt, doc2.txt into ./source/
````

2. Ensure `ontomi.ttl` exists in the project root (for S2+).

3. Run the batch:

```bash
python run_batch.py
```

Outputs will be created under `./output/`.

---

## Step-by-step overview

### S1 — Ingest (JSON)

* Reads the `.txt`
* Cleans and segments text into paragraphs/fragments
* Assigns metadata such as:

  * paragraph index `k`
  * language detection (LID)
  * whether a paragraph is **title/abstract** vs **body**
* Writes:

  * `s1_output.json`

### S2 — Instantiate fragments into RDF (TTL)

* Converts `s1_output.json` into RDF instances:

  * Document + Fragment individuals
  * Labels, identifiers, relationships (`hasPart`, etc.)
* Writes:

  * `instances_fragments.ttl`

### S3 — Evidences (TTL + JSON payload)

* Uses your `evidences_api.py` to annotate each fragment and create evidence individuals:

  * Keyword / ContextObject / DiscursiveStrategy
* Optionally links evidences to intelligences (`onto:evokesIntelligence`)
* Writes:

  * `instances_fragments_evidences.ttl`
  * `evidences_payload.json` (used by S4 for scoring)

### S4 — Activations (TTL + scores JSON)

* Computes activation scores `a_{k,j}` per fragment/intelligence from evidences
* Picks primary intelligence per fragment, and links “secondary” if above threshold (theta)
* Writes:

  * `scores_by_fragment.json`
  * `instances_fragments_activations.ttl`

### S5 — Profile vector (TTL)

* Builds MI profile vectors from activations:

  * per document, per fragment, or both
* Supports normalization:

  * L1 / L2 / softmax (with temperature `tau`)
* Optionally writes a percent-string representation (`onto:miVector`)
* Writes:

  * `instances_fragments_profile.ttl`

### S6 — CQ1/CQ2/CQ3 tests (text outputs)

Runs internal SPARQL-based queries and saves the results:

* `cq1_evoked.txt` — which intelligences are evoked, based on profile vectors
* `cq2_elements_by_intel.txt` — contributing elements grouped by intelligence
* `cq3_top_intelligence.txt` — primary vs secondary intelligences and scores

### S7 — SHACL validation

Validates the final RDF graph using SHACL constraints (and optional inference).

Outputs:

* `shacl_report.ttl`
* `shacl_report.txt`

---

## CLI usage

### Basic

```bash
python run_batch.py --source-dir source --out-dir output
```

### Turn steps on/off

* `--no-s2` : disable S2 (and therefore S3..S7)
* `--no-s3` : disable S3 (and therefore S4..S7)
* `--no-s4` : disable S4 (and therefore S5..S7)
* `--no-s5` : disable S5 (and therefore S6..S7)
* `--no-s6` : disable S6 (and therefore S7)
* `--no-s7` : disable S7

---

## Full parameter reference (grouped)

### Paths

* `--source-dir` (default: `source`)
  Folder containing `.txt` files.
* `--out-dir` (default: `output`)
  Output root folder.

---

### S2 — RDF instantiation

* `--onto` (default: `ontomi.ttl`)
  Path to OntoMI ontology (TTL).
* `--base-ns` (default: `None`)
  Base namespace for instances; if omitted, uses `doc_id#`.
* `--s2-graph` (default: `full`)
  Graph content mode:

  * `full` → ontology + instances in one graph
  * `instances` → only instances
  * `instances+imports` → instances plus `owl:imports`

---

### S3 — Evidences

* `--s3-no-llm`
  Tries to run S3 without an LLM (only works if your `evidences_api.py` supports it).
* `--s3-no-evokes`
  Do not create `onto:evokesIntelligence` links for evidences.

---

### S4 — Activations (threshold + weights)

* `--theta` (default: `0.75`)
  Relative threshold to link secondary intelligences per fragment.
* `--w-keyword` (default: `1.00`)
  Weight for Keyword evidences.
* `--w-context` (default: `1.25`)
  Weight for ContextObject evidences.
* `--w-strategy` (default: `1.10`)
  Weight for DiscursiveStrategy evidences.

---

### S5 — MI profile vector

* `--s5-scope` (default: `both`)

  * `document` → only doc-level vector
  * `fragment` → only fragment-level vectors
  * `both` → both doc + fragment vectors
* `--s5-norm` (default: `l1`)

  * `l1`, `l2`, or `softmax`
* `--s5-tau` (default: `1.0`)
  Softmax temperature (only if `--s5-norm=softmax`).
* `--alpha-title` (default: `1.30`)
  Weight multiplier for title/abstract paragraphs.
* `--alpha-body` (default: `1.00`)
  Weight multiplier for body paragraphs.
* `--s5-vec-places` (default: `4`)
  Decimal places in vector scores.
* `--s5-no-mi-vector-string`
  Disables writing the percent string (`onto:miVector`).

---

### S7 — SHACL

* `--s7-inference` (default: `rdfs`)
  Inference mode for pySHACL:

  * `none`
  * `rdfs`
  * `owlrl`

---

### S1 — Ingest core options

* `--lang-hint` (default: `None`, choices: `pt|en|es`)
  Restricts supported languages to a single one.
* `--cleaners` (default: `ftfy,clean_text,pdf_breaks`)
  CSV list of cleaners (passed into `intelli3text` pipeline).
* `--lid-primary` (default: `fasttext`)
  Primary language ID engine.
* `--lid-fallback` (default: `none`)
  Fallback LID engine (`cld3`, etc.) or `none`.
* `--languages` (default: `pt,en,es`)
  Supported languages (CSV).
* `--nlp-size` (default: `lg`, choices: `lg|md|sm`)
  spaCy model preference.

---

### S1 — segmentation and fallback splitting

* `--paragraph-min-chars` (default: `30`)
  Minimum paragraph length.
* `--lid-min-chars` (default: `60`)
  Minimum paragraph length to run LID.
* `--split-max-chars` (default: `900`)
  Target chunk size when resegmenting.
* `--split-min-chars` (default: `120`)
  Minimum chunk size (small chunks may be merged).
* `--no-resegment`
  Disables fallback resegmentation when a single huge paragraph is detected.

---

### S1 — title/abstract detection

* `--force-title`
  Forces marking some paragraph as title/abstract.
* `--title-scan-k` (default: `3`)
  How many first paragraphs to scan as candidate title.
* `--title-max-chars` (default: `160`)
  Max characters allowed to classify a paragraph as title.

---

## Examples

### 1) Run **only S1** (ingest JSON)

```bash
python run_batch.py --no-s2
```

### 2) Run S1 + S2 (stop before evidences)

```bash
python run_batch.py --no-s3
```

### 3) Full pipeline, but disable LLM in S3

```bash
python run_batch.py --s3-no-llm
```

### 4) Increase secondary linking strictness (fewer secondaries)

```bash
python run_batch.py --theta 0.90
```

### 5) Softmax normalization for MI vectors

```bash
python run_batch.py --s5-norm softmax --s5-tau 0.7
```

### 6) Use an instances-only RDF graph (lighter TTL)

```bash
python run_batch.py --s2-graph instances
```

### 7) Fully

```bash
python3 run_batch.py --source-dir source --out-dir output --lang-hint pt   --cleaners "ftfy,clean_text,pdf_breaks"   --lid-primary fasttext --lid-fallback none --languages "pt,en,es"   --nlp-size md --paragraph-min-chars 60 --lid-min-chars 60   --split-max-chars 900 --split-min-chars 250   --force-title --title-scan-k 5 --title-max-chars 160   --onto ontomi.ttl --s2-graph instances   --theta 0.75 --w-keyword 1.00 --w-context 1.25 --w-strategy 1.10   --s5-scope both --s5-norm l1 --s5-tau 1.0   --alpha-title 1.30 --alpha-body 1.00 --s5-vec-places 4   --s7-inference rdfs
```

---

## Outputs and telemetry

For each document:

* `run_log.json` contains durations and per-step metrics
* `batch_report.json` aggregates outcomes for the whole run

Failures:

* If processing a file fails, its output folder will contain `s1_error.txt` and the batch continues.

---

## Troubleshooting

### “ontologia não encontrada”

If S2 is enabled, `--onto` must exist (default: `ontomi.ttl`).

### S3 fails with LLM / Ollama errors

If your `evidences_api.py` uses an LLM:

* Ensure the LLM runtime is up
* Ensure the target model is pulled/available
* Or run with `--s3-no-llm` (only if your evidences implementation supports it)

### SHACL doesn’t conform

Check `shacl_report.txt` and `shacl_report.ttl` in the document folder.

---



