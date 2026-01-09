# S4 — Activations

## Purpose
Aggregates evidences into activation scores per fragment and intelligence.

## What it does
- Computes activation values a_{k,j}
- Determines primary and secondary intelligences per fragment
- Applies role weights and thresholds

## Inputs
- `instances_fragments_evidences.ttl`
- `evidences_payload.json`

## Outputs
- `instances_fragments_activations.ttl`
- `scores_by_fragment.json`

## Key Parameters
- `--theta`
- `--w-keyword`
- `--w-context`
- `--w-strategy`
