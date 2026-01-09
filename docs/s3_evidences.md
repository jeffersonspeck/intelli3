# S3 — Evidences

## Purpose
Extracts evidence elements from each fragment and associates them with Multiple Intelligences.

## What it does
- Extracts candidate terms (YAKE)
- Classifies them using LLM (optional)
- Produces Keyword, ContextObject, and DiscursiveStrategy evidences
- Optionally links evidences to intelligences

## Inputs
- `s1_output.json`
- `instances_fragments.ttl`

## Outputs
- `instances_fragments_evidences.ttl`
- `evidences_payload.json`

## Key Parameters
- `--s3-no-llm`
- `--s3-no-evokes`

## Notes
This is the main cognitive annotation step of Intelli3.
