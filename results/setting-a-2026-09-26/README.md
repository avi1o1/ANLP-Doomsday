# Setting A results — 2026-09-26

Curated results from `research-v5`, originally produced at
`/ssd_scratch/anirudh.sankar/output/research-v5` on gnode069.

## Contents

- `research-v5/report/`: retrieval tables, baselines, Indic macro-averages,
  predictor metrics, resources, a figure and completion inventory.
- `research-v5/analysis/`: aggregate diagnostic rows and component ablations.
- `research-v5/predictor/`: small predictor checkpoints and validation summaries.
  `validation_summary.json` retains fold metrics, aggregate metrics, ordering and
  uncertainty from the full validation output, omitting per-example predictions.
- `parts/manifest.json`: sizes and SHA-256 checksums of the 14 downloaded archives.
- `verification.json`: records successful archive verification and validation of
  all 182 per-query outcome matrices before curating this snapshot.

The primary run covers 182 representation–collection rows with eight scoring
configurations each, plus 26 baseline evaluations. Additional experiment families
in the expanded manifest are not complete; see the inventory and report.

## Local analysis payloads

Archives, per-query outcome matrices, duplicate evaluation records, diagnostic
caches and full validation predictions remain untracked under
`context/results-2026-09-26/`. They are not required to read the committed summaries.
A fresh clone contains the curated evidence, not the full analysis snapshot.
The original outputs remain on cluster scratch; the separate storage-capped backup
is independent of this directory. No secrets, embeddings or indexes are included.

The archive manifests refer to the original downloaded layout. Extracting the
archives restores that full layout, including copies of these curated files.
