# S2 — RDF Instantiation

## Purpose
Transforms the structured JSON from S1 into RDF/OWL instances aligned with the OntoMI ontology.

## What it does
- Creates Document and ExplanationFragment individuals
- Links fragments to documents
- Assigns identifiers, labels, and structural relations

## Inputs
- `s1_output.json`
- `ontomi.ttl` ontology

## Outputs
- `instances_fragments.ttl`

## Key Parameters
- `--onto`
- `--base-ns`
- `--s2-graph`

## Notes
This stage materializes the conceptual layer used by all semantic reasoning steps.
