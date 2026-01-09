# S7 — SHACL Validation

## Purpose
Validates the generated RDF graph against OntoMI SHACL constraints.

## What it does
- Runs pySHACL validation
- Applies optional inference (RDFS / OWL RL)
- Generates human-readable and TTL reports

## Inputs
- `instances_fragments_profile.ttl`
- `ontomi.ttl`

## Outputs
- `shacl_report.ttl`
- `shacl_report.txt`

## Key Parameters
- `--s7-inference`

## Notes
Ensures structural and semantic consistency of the knowledge graph.
