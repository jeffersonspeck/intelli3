# S5 — Profile Vector

## Purpose
Builds and normalizes Multiple Intelligence profile vectors.

## What it does
- Aggregates fragment activations at document and/or fragment level
- Applies alpha weighting (title vs body)
- Normalizes vectors (L1, L2, Softmax)
- Optionally writes percent string representation

## Inputs
- `instances_fragments_activations.ttl`
- `s1_output.json`

## Outputs
- `instances_fragments_profile.ttl`

## Key Parameters
- `--s5-scope`
- `--s5-norm`
- `--s5-tau`
- `--alpha-title`
- `--alpha-body`
