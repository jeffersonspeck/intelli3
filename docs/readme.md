# Intelli3 — Documentation Index

This directory contains the **technical documentation** for the Intelli3 pipeline.
Each file describes one stage of the batch process, its purpose, inputs, outputs,
parameters, and its role in the overall architecture.

Intelli3 is a **research-oriented, ontology-driven pipeline** designed to analyze
educational texts and generate **explainable Multiple Intelligences (MI) profiles**,
combining symbolic modeling (RDF/OWL), statistical signals, and optional LLM-based
annotation.

---

## Pipeline Overview

The Intelli3 pipeline is executed in batch mode via `run_batch.py` and is organized
into **seven sequential stages (S1–S7)**.  
Each stage produces explicit artifacts that can be inspected independently.

S1 → S2 → S3 → S4 → S5 → S6 → S7

> Dependency rule: if a stage is disabled, all subsequent stages are automatically skipped.

---

## Documentation Structure

### S1 — Ingest
**File:** [`s1_ingest.md`](s1_ingest.md)

- Raw text ingestion and normalization
- Paragraph/fragment segmentation
- Language identification (LID)
- Title/abstract detection
- Output: structured JSON (`s1_output.json`)

This stage transforms unstructured text into a controlled analytical unit.

---

### S2 — RDF Instantiation
**File:** [`s2_instantiate.md`](s2_instantiate.md)

- Instantiates Documents and Fragments as RDF individuals
- Aligns data with the OntoMI ontology
- Establishes structural relations (`hasFragment`, `isPartOf`)
- Output: `instances_fragments.ttl`

This stage creates the **symbolic knowledge layer** used by all semantic reasoning.

---

### S3 — Evidences
**File:** [`s3_evidences.md`](s3_evidences.md)

- Extracts semantic evidences from fragments
- Evidence roles:
  - `Keyword`
  - `ContextObject`
  - `DiscursiveStrategy`
- Optional LLM-based MI attribution
- Output:
  - `instances_fragments_evidences.ttl`
  - `evidences_payload.json`

This is the **core semantic annotation stage** of Intelli3.

---

### S4 — Activations
**File:** [`s4_activations.md`](s4_activations.md)

- Aggregates evidences into activation scores \( a_{k,j} \)
- Applies role-specific weights
- Determines primary and secondary intelligences per fragment
- Output:
  - `instances_fragments_activations.ttl`
  - `scores_by_fragment.json`

This stage operationalizes the transition from **evidence** to **cognitive activation**.

---

### S5 — Profile Vector
**File:** [`s5_profile_vector.md`](s5_profile_vector.md)

- Builds MI profile vectors (fragment-level and/or document-level)
- Supports normalization strategies:
  - L1
  - L2
  - Softmax (with temperature)
- Optional string representation (`onto:miVector`)
- Output: `instances_fragments_profile.ttl`

This stage produces the **final explainable MI representation**.

---

### S6 — Competency Questions (CQ)
**File:** [`s6_tests.md`](s6_tests.md)

Runs SPARQL-based diagnostic queries:

- **CQ1:** Which intelligences are evoked (by fragment)?
- **CQ2:** Which elements contribute to each intelligence?
- **CQ3:** What is the primary vs secondary intelligence ranking?

Outputs:
- `cq1_evoked.txt`
- `cq2_elements_by_intel.txt`
- `cq3_top_intelligence.txt`

This stage supports **interpretability, debugging, and research validation**.

---

### S7 — SHACL Validation
**File:** [`s7_shacl_validate.md`](s7_shacl_validate.md)

- Validates the final RDF graph against OntoMI SHACL constraints
- Supports inference modes (none, RDFS, OWL RL)
- Outputs:
  - `shacl_report.ttl`
  - `shacl_report.txt`

This stage ensures **structural and semantic consistency** of the knowledge graph.

---

## Outputs and Traceability

For each processed document, Intelli3 produces:

- A dedicated output folder: `output/<doc_slug>/`
- A detailed execution log: `run_log.json`

For the entire batch:

- `batch_report.json` with aggregated metrics and execution summary

This design ensures **full traceability and reproducibility**, which is essential
for experimental and academic use.

---

## Research Context

The Intelli3 pipeline is part of a **Master’s research project in Computer Science**
focused on:

- Explainable semantic modeling of educational content
- Operationalization of the Theory of Multiple Intelligences
- Integration of ontologies, symbolic reasoning, and LLMs

The documentation in this folder is intended to support:
- Scientific reproducibility
- Code inspection and extension
- Academic evaluation (papers, dissertation, peer review)

---

## Entry Points

- Batch execution: `run_batch.py`
- Ontology definition: `ontomi.ttl`
- Evidence logic: `evidences_api.py`

For usage instructions and CLI parameters, refer to the **main repository README**.

---