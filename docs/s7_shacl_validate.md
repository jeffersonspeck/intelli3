# S7 — SHACL Validation

## Summary
S7 validates the generated RDF graph against OntoMI SHACL constraints to ensure
structural and semantic consistency.

## Inputs

- `instances_fragments_profile.ttl`
- `ontomi.ttl`

## Outputs

- `shacl_report.ttl`
- `shacl_report.txt`

## How it works

1. Runs pySHACL over the final RDF graph.
2. Applies the configured inference mode.
3. Writes a TTL report and a human-readable text summary.

## Key parameters

- `--s7-inference`: `none`, `rdfs`, or `owlrl`.

## Notes and tips

- If validation fails, inspect the text report first for human-readable details.
- SHACL is the last stage, so failures do not affect earlier artifacts.
