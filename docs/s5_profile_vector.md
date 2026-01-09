# S5 — Profile Vector

## Summary
S5 builds and normalizes Multiple Intelligence (MI) profile vectors at the
fragment and/or document level. It produces the final explainable MI
representation used by downstream analysis and the classifier.

## Inputs

- `instances_fragments_activations.ttl`
- `s1_output.json`

## Outputs

- `instances_fragments_profile.ttl`

## How it works

1. Aggregates activations by fragment or document.
2. Applies title/body weighting (`alpha` factors).
3. Normalizes vectors using L1, L2, or softmax.
4. Optionally writes the `onto:miVector` string representation.

## Key parameters

- `--s5-scope`: `document`, `fragment`, or `both`.
- `--s5-norm`: `l1`, `l2`, or `softmax`.
- `--s5-tau`: temperature for softmax normalization.
- `--alpha-title` / `--alpha-body`: weighting for title vs body.
- `--s5-vec-places`: decimal places written in vector scores.
- `--s5-no-mi-vector-string`: disable string representation output.

## Notes and tips

- If you need only document-level vectors, set `--s5-scope document`.
- Use softmax normalization to emphasize dominant intelligences.
