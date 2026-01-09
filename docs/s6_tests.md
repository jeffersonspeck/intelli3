# S6 — Competency Questions (CQ)

## Summary
S6 runs diagnostic SPARQL queries to inspect semantic results and provide
interpretability during research and debugging.

## Inputs

- `instances_fragments_profile.ttl`

## Outputs

- `cq1_evoked.txt`
- `cq2_elements_by_intel.txt`
- `cq3_top_intelligence.txt`

## Queries executed

- **CQ1:** Evoked intelligences by fragment.
- **CQ2:** Evidence grouped by intelligence.
- **CQ3:** Primary vs secondary intelligence ranking.

## Notes and tips

- Use these outputs to validate whether evidences and activations align with
  expected theoretical behavior.
- Outputs are plain text, so they are easy to inspect or diff across runs.
