# S2 — RDF Instantiation

## Summary
S2 transforms the structured JSON from S1 into RDF/OWL instances aligned with the
OntoMI ontology. It builds the symbolic layer used by semantic reasoning steps.

## Inputs

- `s1_output.json`
- `ontomi.ttl` ontology (or another file passed via `--onto`)

## Outputs

- `instances_fragments.ttl`

## How it works

1. Creates `Document` and `ExplanationFragment` individuals.
2. Links fragments to documents using structural relations.
3. Assigns identifiers, labels, and metadata for downstream stages.

## Key parameters

- `--onto`: path to the ontology TTL file.
- `--base-ns`: base namespace for generated instances.
- `--s2-graph`: output graph mode (`full`, `instances`, `instances+imports`).

## Notes and tips

- Ensure the ontology file is present before running S2.
- Use `instances` if you want a lighter TTL with only instance data.
