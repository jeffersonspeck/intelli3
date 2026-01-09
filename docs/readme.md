# Intelli3 Documentation Index

This directory contains the **technical documentation** for the Intelli3 pipeline.
Each file describes a pipeline stage, its purpose, inputs, outputs, and how it fits
into the overall architecture.

## Pipeline Overview

The Intelli3 pipeline runs in batch mode via `run_batch.py` and is organized into
**seven sequential stages (S1–S7)**. Each stage produces artifacts that can be
inspected independently.

S1 → S2 → S3 → S4 → S5 → S6 → S7

**Dependency rule:** if a stage is disabled, all subsequent stages are skipped.

## Stage Documentation

| Stage | File | Focus |
| --- | --- | --- |
| S1 — Ingest | [`s1_ingest.md`](s1_ingest.md) | Input ingestion, segmentation, LID, title detection |
| S2 — RDF Instantiation | [`s2_instantiate.md`](s2_instantiate.md) | RDF individuals aligned to OntoMI |
| S3 — Evidences | [`s3_evidences.md`](s3_evidences.md) | Evidence extraction, optional LLM signals |
| S4 — Activations | [`s4_activations.md`](s4_activations.md) | Evidence-to-activation scoring |
| S5 — Profile Vector | [`s5_profile_vector.md`](s5_profile_vector.md) | MI vector aggregation/normalization |
| S6 — Tests (CQ) | [`s6_tests.md`](s6_tests.md) | SPARQL-based diagnostics |
| S7 — SHACL Validation | [`s7_shacl_validate.md`](s7_shacl_validate.md) | Graph validation with SHACL |

## Outputs and Traceability

For each processed document, Intelli3 produces:

- A dedicated output folder: `output/<doc_slug>/`
- A detailed execution log: `run_log.json`

For the entire batch:

- `batch_report.json` with aggregated metrics and execution summary

This design ensures **traceability and reproducibility**, which is essential for
experimental and academic use.

## Research Context

The Intelli3 pipeline is part of a **Master’s research project in Computer Science**
focused on:

- Explainable semantic modeling of educational content
- Operationalization of the Theory of Multiple Intelligences
- Integration of ontologies, symbolic reasoning, and LLMs

## Entry Points

- Batch execution: `run_batch.py`
- Ontology definition: `ontomi.ttl`
- Evidence logic: `evidences_api.py`

For usage instructions and CLI parameters, refer to the **main repository README**.
