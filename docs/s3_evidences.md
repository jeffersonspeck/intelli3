# S3 — Evidences

## Summary
S3 extracts evidence elements from each fragment and associates them with
Multiple Intelligences. It is the main semantic annotation stage of Intelli3.

## Inputs

- `s1_output.json`
- `instances_fragments.ttl`

## Outputs

- `instances_fragments_evidences.ttl`
- `evidences_payload.json`

## How it works

1. Extracts candidate terms (e.g., YAKE keywords).
2. Classifies evidences using local logic or an LLM (optional).
3. Creates evidence types:
   - `Keyword`
   - `ContextObject`
   - `DiscursiveStrategy`
4. Optionally links evidences to intelligences (`onto:evokesIntelligence`).

## Key parameters

- `--s3-no-llm`: disable LLM calls (only if the implementation provides a fallback).
- `--s3-no-evokes`: generate evidences without `evokesIntelligence` links.

## Notes and tips

- LLM usage depends on how `evidences_api.py` is configured.
- Inspect `evidences_payload.json` to debug evidence extraction.
