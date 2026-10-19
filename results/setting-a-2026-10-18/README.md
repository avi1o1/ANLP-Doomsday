# Setting A results, MS MARCO budget sweep and SPLADE-v3, 18 October 2026

What the cluster added after the 29 September workstation run, and the combined analysis.
Everything here comes from jobs on the Ada cluster (gnode060, RTX 2080 Ti); the
workstation rows are unchanged.

## Contents

| Path | What it is |
|---|---|
| `analysis/msmarco_rows.json` | The 126 MS MARCO rows, diagnosed with 10,000 bootstrap draws: 42 primary rows (q32_d128, from the cluster run research-v5) and 84 new budget rows (q16_d64 and q64_d256) from the sweep. |
| `analysis/msmarco_component_ablations.json` | Their 126 component ablations (3 per primary row). |
| `analysis/rows.json`, `analysis/component_ablations.json` | The union with the 29 September workstation rows (`results/setting-a-2026-09-29/analysis/`): 1,352 rows, 182 primary. Holm correction is redone over the 182 primary rows and over 182 rows per component (`scripts/merge_analysis_rows.py`). |
| `exploratory/shapley_*` | Shapley attribution over the 182 primary rows (13 collections). |
| `splade/splade_v3.csv` | SPLADE-v3 (`naver/splade-v3`, revision fdfeceb) as an external reference, full corpus, 13 collections. |

## MS MARCO budget sweep

The 84 rows (three MS MARCO collections, 14 representations, two budgets) were scored on
the cluster from the existing MS MARCO encodings, stored at the 256-feature cap, by
re-indexing at each budget. Mean nDCG@10 over the 14 representations:

| Collection | Budget | All off | All on | All on is best of 8 |
|---|---|---:|---:|---:|
| msmarco_dev | q16_d64 | 0.010 | 0.082 | 9/14 |
| | q32_d128 | 0.013 | 0.130 | 6/14 |
| | q64_d256 | 0.015 | 0.168 | 6/14 |
| trec_dl_2019 | q16_d64 | 0.030 | 0.172 | 8/14 |
| | q32_d128 | 0.036 | 0.255 | 7/14 |
| | q64_d256 | 0.040 | 0.323 | 5/14 |
| trec_dl_2020 | q16_d64 | 0.029 | 0.160 | 7/14 |
| | q32_d128 | 0.035 | 0.251 | 7/14 |
| | q64_d256 | 0.037 | 0.307 | 4/14 |

The three collections share one corpus, so their diagnostics are identical and their rows
are not independent. As the budget grows, all on improves, while the number of
representations where it is the best of the eight configurations falls on all three
collections (9 to 6, 8 to 5, 7 to 4 between q16_d64 and q64_d256).

## Combined significance and attribution

Holm-corrected comparisons significant at 0.05, all on against all off: 99 of 140
workstation primary rows (unchanged), and 134 of 182 with the 42 MS MARCO primary rows.
With 182 comparisons the floor at 10,000 draws is 182/10001 = 0.018.

Shapley shares over the 182 primary rows: IDF 10.1%, saturation 61.8%, length 28.2%.
Collection statistics (IDF plus length) are 38.3% of the gain (95% interval 36.1 to 40.7%,
resampling collections). This agrees with the cluster-only 182-row figure in
`results/setting-a-2026-09-26`.

## SPLADE-v3

nDCG@10 on full corpora, queries with judgments only:

| Collection | nDCG@10 | Collection | nDCG@10 |
|---|---:|---|---:|
| msmarco_dev | 0.470 | scifact | 0.717 |
| trec_dl_2019 | 0.722 | trec_covid | 0.730 |
| trec_dl_2020 | 0.751 | arguana | 0.489 |
| fiqa | 0.380 | nfcorpus | 0.364 |
| scidocs | 0.156 | limit | 0.447 |
| miracl_hi | 0.008 | miracl_bn | 0.000 |
| miracl_te | 0.001 | | |

The MIRACL collections are near zero because SPLADE-v3's English vocabulary does not
tokenise Hindi, Bengali or Telugu. These are the plain-retrieval runs. SPLADE scored under
all eight IDF, saturation and length combinations, which the Appendix C reversal needs, is
not in this directory (`scripts/export_splade_features.py` and the GPU path in
`src/retrieval/external.py` were written for it).

## Not done in this directory

- `scripts/test_hypotheses.py` and `scripts/predictor_study.py` need label-free index
  diagnostics (`mechanism_diagnostics.py`) and ranking features (`ranking_features.py`)
  for the MS MARCO indexes and rows. These come from the built indexes and stored
  rankings on the cluster and have not been computed, so those results are not updated.
- The adapter validation at scale.

## Provenance

Row sets: workstation run `results/setting-a-2026-09-29/analysis/`; cluster run
research-v5 (42 primary MS MARCO rows); sweep job 23634 on gnode060. Diagnosis job 26434,
`significance.bootstrap_samples: 10000`.
