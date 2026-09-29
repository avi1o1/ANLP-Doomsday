# Setting A results, single-GPU workstation run, 29 September 2026

Curated summaries of the second Setting A run, made on a single-GPU workstation
(one RTX 2000 Ada, eight cores, 30 GB) with `configs/workstation-no-msmarco.yaml`. It covers
every experiment family on the ten collections outside the MS MARCO corpus.

## What the run contains

| Family | Rows |
|---|---:|
| primary | 140 |
| sparsity (budgets q16_d64, q32_d128, q64_d256, and native on three collections) | 462 |
| parameters (k1 and b over nine cells, five collections) | 630 |
| robustness (seeds 1 and 2, five stochastic bases, five collections) | 100 |
| controls (shuffled and uniform document frequency, three collections) | 84 |
| shared_sae | 20 |
| union | 1226 |

Families overlap in their reference cells, so the union is smaller than the sum.

## How it relates to the cluster run

The seed-0 bases are the cluster run's own (`results/setting-a-2026-09-26`), imported
with every checksum, specification and encoder revision checked. The fitting sample
selects the same 200,000 documents (identical ID hash), datasets and E5 revision are
pinned to the cluster's frozen configuration, and BM25 reproduces the cluster's
baselines to seven decimals. Only the seed-1 and seed-2 bases were fitted here.

`replication.csv` compares the 140 primary rows the two runs share. The median absolute
difference in all-on nDCG@10 is 0.0007 (maximum 0.0077), the median margin difference is
0.011, and the margin sign agrees in 136 of 137 rows with defined margins; the one
disagreement is a margin near zero in both runs. The differences are fp16 rounding on a
different GPU. Rows from the two runs describe the same representations, and
the sensitivity families here can be read against the cluster's primary rows.

## Differences from the cluster run

- Significance uses 10,000 bootstrap samples, in a separate `significance` block that
  evaluations are not keyed on. At 2,000 samples Holm correction across a family of 182
  could not fall below 0.091. Here, with 140 comparisons per family, the floor is
  140/10001 = 0.014.
- Rows whose outcome responds to fewer than two of the three switches are held out of
  the predictor fit (19 primary rows, listed in `predictor/excluded_rows.json`); their
  margin is fixed at about 2 by construction.
- Per-query records keep the top 100 of each ranking, compactly. Every metric is still
  computed from the full top 1000 before writing.

## Headline numbers

Holm-corrected comparisons significant at 0.05, out of 140 primary rows:

| Comparison | Significant |
|---|---:|
| all switches on against all off | 99 |
| saturation main effect | 93 |
| length-normalisation main effect | 79 |
| IDF main effect | 56 |

Mean main effects on nDCG@10: saturation +0.065, length +0.032, IDF +0.013.

The pre-registered predictor remains weak. On the primary rows (n = 118 after exclusions) its sign
accuracy equals the majority-sign baseline, 0.924, and its held-out R² is −0.11 across
basis families and 0.14 across corpora. See `report/predictor.csv`. The exploratory
analysis in `exploratory/README.md` attributes the gain to each switch and builds a
label-free predictor that does better.

The dense encoder baselines, at the same encoder revision as the cluster run, are in
`report/baselines.json` for all ten collections. The final report lists no incomplete
or failed artifacts.

## Not included

- The three MS MARCO collections. Their q32_d128 rows are in the cluster run. Their
  q16_d64 and q64_d256 rows were being built on the cluster when every cluster job was
  cancelled for maintenance; see `context/msmarco-sweep-2026-09-29/`.

Per-query records, encodings and indexes remain on the workstation and are not included.
