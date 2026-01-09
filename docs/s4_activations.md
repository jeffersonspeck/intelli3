# S4 — Activations

## Summary
S4 aggregates evidences into activation scores per fragment and intelligence.
It converts symbolic evidences into numeric signals used by the profile vector.

## Inputs

- `instances_fragments_evidences.ttl`
- `evidences_payload.json`

## Outputs

- `instances_fragments_activations.ttl`
- `scores_by_fragment.json`

## How it works

1. Computes activation values `a_{k,j}` for fragment `k` and intelligence `j`.
2. Applies role-specific weights for evidence types.
3. Determines primary and secondary intelligences per fragment.

## Key parameters

- `--theta`: threshold for secondary intelligences.
- `--w-keyword`: weight for `Keyword` evidences.
- `--w-context`: weight for `ContextObject` evidences.
- `--w-strategy`: weight for `DiscursiveStrategy` evidences.

## Notes and tips

- Increase `--theta` for fewer secondary intelligences.
- Check `scores_by_fragment.json` to inspect raw activation values.
