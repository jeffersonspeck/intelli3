# S1 — Ingest

## Summary
S1 ingests raw inputs from `source/` and converts them into a structured JSON
representation used by later stages. It handles text cleaning, segmentation,
language detection, and title/abstract detection.

## Inputs

- Raw files under `source/` (typically `.txt`).
- CLI parameters passed via `run_batch.py` (cleaners, segmentation rules, LID, etc.).
- Optional local module: `intelli3text` (text processing helpers).

## Outputs

- `s1_output.json` containing:
  - `doc_id`
  - a list of paragraphs/fragments with `k`, text, language, and metadata

## How it works

1. Reads each input file and normalizes encoding/whitespace.
2. Applies configured cleaner passes.
3. Segments text into paragraphs/fragments.
4. Detects language per fragment (or uses `--lang-hint`).
5. Identifies title/abstract candidates.

## Key parameters

- `--cleaners`: comma-separated cleaners (e.g., `ftfy,clean_text,pdf_breaks`).
- `--languages`: list of language codes used by LID (e.g., `pt,en,es`).
- `--lang-hint`: force a single language to speed up LID.
- `--paragraph-min-chars`: minimum length for a valid paragraph.
- `--split-max-chars` / `--split-min-chars`: fallback splitting for long paragraphs.
- `--force-title`: always mark initial paragraph(s) as title/abstract.

## Notes and tips

- S1 is mandatory; all later stages depend on its output.
- If LID is unstable, try `--lang-hint` or increase `--lid-min-chars`.
- For PDF-derived text, include `pdf_breaks` in `--cleaners`.
