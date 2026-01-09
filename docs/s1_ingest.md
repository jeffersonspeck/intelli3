# S1 — Ingest

## Purpose
The S1 stage is responsible for ingesting raw input texts from the `source/` directory and converting them into a structured JSON representation.

## What it does
- Reads files
- Cleans text using configurable cleaners
- Segments text into paragraphs/fragments
- Detects language (LID)
- Identifies title/abstract vs body paragraphs
- Assigns a stable fragment index `k`

## Inputs
- Raw file
- Parameters passed via `run_batch.py` (cleaners, LID, segmentation rules)

## Outputs
- `s1_output.json` containing:
  - `doc_id`
  - paragraph list with `k`, text, language, and metadata

## Key Parameters
- `--cleaners`
- `--languages`
- `--lang-hint`
- `--paragraph-min-chars`
- `--split-max-chars`
- `--force-title`

## Role in the pipeline
S1 is mandatory. All subsequent stages depend on its output.
