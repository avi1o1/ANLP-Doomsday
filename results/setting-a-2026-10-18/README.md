# Setting A results, MS MARCO budget sweep and SPLADE-v3, 18 October 2026

What the cluster added after the 29 September workstation run, and the combined analysis.
Everything here comes from jobs on the Ada cluster (gnode060, RTX 2080 Ti); the
workstation rows are unchanged.

## Contents

| Path | What it is |
|---|---|
| `analysis/msmarco_rows.json` | The 126 MS MARCO rows, diagnosed with 10,000 bootstrap draws: 42 primary rows (q32_d128, from the cluster run research-v5) and 84 new budget rows (q16_d64 and q64_d256) from the sweep. |
| `analysis/msmarco_component_ablations.json` | Their 126 component ablations (3 per primary row). |
| `analysis/rows.json`, `analysis/component_ablations.json` | the union with the 29 September workstation rows (`results/setting-a-2026-09-29/analysis/`) and the six shared-SAE rows: 1,358 rows, 182 primary. Holm correction is redone over the 182 primary rows and over 182 rows per component (`scripts/merge_analysis_rows.py`). |
| `exploratory/shapley_*` | Shapley attribution over the 182 primary rows (13 collections). |
| `exploratory/confirmatory_tests.json`, `predictor_study*.json` | Hypothesis tests and predictor study re-run with the MS MARCO rows. |
| `exploratory/mechanism_diagnostics.json`, `ranking_features.json` | Workstation files with the MS MARCO indexes and rows appended. |
| `analysis/msmarco_shared_sae_rows.json` | The six shared-SAE rows for MS MARCO. |
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
tokenise Hindi, Bengali or Telugu. These are the plain-retrieval runs.

## Appendix C reversal on SPLADE-v3

The stored SPLADE-v3 document features were scored under all eight IDF, saturation and
length combinations (`splade/eight_configs/`, `splade/splade_v3_eight_configs.csv`;
`scripts/export_splade_features.py`, GPU scoring in `src/retrieval/external.py`). All off
is the plain inner product; all on is the full correction. The all-off column reproduces
the plain SPLADE run above to three decimals on every collection checked. The comparison is
conceptual: it is a different backbone and protocol from Clavie et al.'s Appendix C.

| Collection | SPLADE off | SPLADE on | E5 mean off (14 reps) | E5 mean on |
|---|---:|---:|---:|---:|
| arguana | 0.489 | 0.563 | 0.021 | 0.130 |
| fiqa | 0.380 | 0.318 | 0.013 | 0.084 |
| limit | 0.447 | 0.667 | 0.001 | 0.004 |
| msmarco_dev | 0.470 | 0.382 | 0.013 | 0.130 |
| nfcorpus | 0.364 | 0.356 | 0.064 | 0.115 |
| scidocs | 0.156 | 0.154 | 0.009 | 0.044 |
| scifact | 0.717 | 0.716 | 0.088 | 0.197 |
| trec_covid | 0.731 | 0.583 | 0.135 | 0.246 |
| trec_dl_2019 | 0.722 | 0.680 | 0.036 | 0.255 |
| trec_dl_2020 | 0.751 | 0.674 | 0.035 | 0.251 |
| mean of these 10 | 0.523 | 0.509 | 0.042 | 0.146 |

MIRACL is left out of the mean (all scores near zero for SPLADE). On the E5 features the
correction raises every collection's mean. On SPLADE-v3 the plain inner product is as good
or better on 8 of the 10, and the correction helps only on arguana and limit. The p-values
below are uncorrected bootstrap values at 2,000 draws, taken from each collection's own
all-on against all-off comparison: all on is significantly worse (p < 0.05) on fiqa,
msmarco_dev, trec_covid and trec_dl_2020, significantly better on arguana and limit, and
not distinguishable on the other seven (including the three MIRACL collections). The
direction matches the reversal Appendix C reports for SPLADE-v3. The size does not: the
10-collection mean gap is 0.014 here, against 0.113 in their numbers.

## Shared-SAE rows for MS MARCO

The six shared-SAE rows the cluster run never made (`analysis/msmarco_shared_sae_rows.json`,
appended to `analysis/rows.json`, now 1,358 rows): the SAE dictionary fitted on tokens, applied
at pooled and token granularity, q32_d128. They are outside the primary family, as in the
workstation run, so the Holm correction above is unchanged.

| Collection | Granularity | All off | All on |
|---|---|---:|---:|
| msmarco_dev | pooled | 0.171 | 0.303 |
| | token | 0.021 | 0.318 |
| trec_dl_2019 | pooled | 0.354 | 0.534 |
| | token | 0.109 | 0.604 |
| trec_dl_2020 | pooled | 0.312 | 0.517 |
| | token | 0.076 | 0.556 |

All on beats all off in all six rows (margin 2.35 to 2.44, bootstrap p at its floor of 1e-4).

## Confirmatory tests and predictor study with MS MARCO

Label-free index diagnostics (`exploratory/mechanism_diagnostics.json`, 42 MS MARCO indexes,
copied to the three collections that share the corpus) and ranking features
(`exploratory/ranking_features.json`, 126 MS MARCO rows) were computed on the cluster and
appended to the workstation files. The ranking features use the top 100 of each stored
ranking, as the workstation and sweep rows store, because the original cluster rows keep
1,000 and the score-shape features depend on depth (`STORED_DEPTH` in
`scripts/ranking_features.py`).

- `exploratory/confirmatory_tests.json`: H1, H2, H3, H4, H5 and H6 all still pass, with the same
  values as on 29 September. They are computed on the parameter-sweep collections, which
  MS MARCO is not part of, so nothing was expected to change.
- `exploratory/predictor_study*.json` (bm25, dense and fused references): refitted with the
  MS MARCO rows included, which changes both the training rows and the held-out sets. The
  script's summary line for the BM25 reference and ridge regression on pseudo-labels, then
  the same on rankings alone:

```
pseudo-labels / ridge   family +0.53/+0.76/0.78 | corpus +0.44/+0.73/0.77 | other budgets +0.74/+0.80/0.80 | other seeds +0.53/+0.77/0.85 | other parameters +0.59/+0.62/0.72
rankings / ridge        family -0.02/+0.29/0.63 | corpus +0.06/+0.34/0.66 | other budgets +0.17/+0.46/0.69 | other seeds +0.27/+0.41/0.69 | other parameters +0.12/+0.33/0.61
```

  The 29 September values for the same lines were `+0.38/+0.58/0.75`, `+0.52/+0.63/0.71`, ...
  (pseudo-labels, family and corpus) and `-0.14/-0.07/0.55`, `-0.17/+0.12/0.58` (rankings).
  The pseudo-label tier is still the only one with positive held-out R-squared on every axis.
  The movement is not uniform (corpus held-out falls from +0.52 to +0.44 on the first
  number), and no claim about the direction of the change is made here.

## Not done in this directory

- The adapter results (Settings B and C) are not in this directory.
- Nothing here has been committed to the repository.

## Provenance

Shared-SAE rows: jobs 27744 to 27747 on gnode060; ranking features jobs 27733 and 29906; index
diagnostics job 27820.

Row sets: workstation run `results/setting-a-2026-09-29/analysis/`; cluster run
research-v5 (42 primary MS MARCO rows); sweep job 23634 on gnode060. Diagnosis job 26434,
`significance.bootstrap_samples: 10000`.
