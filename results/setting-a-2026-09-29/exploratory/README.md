# Where the gain comes from, and what predicts it

An analysis of the completed Setting A rows beyond the pre-registered margin
regression. Part of it is exploratory and part confirmatory, and each result below says
which. The confirmatory hypotheses were committed in `HYPOTHESES.md` (commit 9df6d58,
29 September 2026 12:55 IST) before any of their tests ran.

## Why the pre-registered target was hard to predict

The primary margin standardises the all-on minus all-off difference by the spread of
quality over all eight configurations. The all-off configuration is a raw inner product
of non-negative features, and in 42% of the 140 primary rows its nDCG@10 is below 0.01.
The margin sits between 1.5 and 3.5 in 83% of the 137 rows where it is defined: it mostly records that
the uncorrected scorer is broken, and leaves little variation for any diagnostic to
explain. All switches on is the best of the eight configurations in only 27 of 140 rows.

## Attribution: who gets credit for the gain

`shapley_summary.json` gives each switch's Shapley value over the eight configurations,
the average of its marginal effect over every order in which the switches can be turned
on. The three values sum exactly to the all-on minus all-off gain. IDF and length
normalisation read collection statistics; saturation reads only the document.

| Run | Rows | IDF | Saturation | Length | Collection statistics (95% interval) |
|---|---:|---:|---:|---:|---|
| Cluster (13 collections, MS MARCO included) | 182 | 10% | 62% | 28% | 38% (36-41%) |
| Workstation (10 collections) | 140 | 10% | 61% | 29% | 39% (36-42%) |

Intervals resample whole collections. The shares agree across the two runs to within a
point. Once saturation is applied, collection statistics add 0.015 nDCG@10 on average
(0.021 on the cluster run), positive in about two thirds of rows. Their share is
largest for the identity and Gaussian bases (62-71%) and smallest for product
quantisation and k-means (9-11%).

The attribution is descriptive and was not pre-registered; its replication across two
independent runs is what supports it.

## The IDF control

The pre-registered controls replace document frequencies with shuffled or uniform
values on ArguAna, NFCorpus and SciFact (42 rows each). IDF's Shapley value is +0.011
with the real frequencies, -0.011 with shuffled ones (real larger in 83% of rows,
one-sided Wilcoxon p = 8.9e-7) and zero with uniform ones by construction. The collection's
frequencies carry information that a random reweighting does not. Its size is small:
all-on nDCG@10 with uniform frequencies is not lower than with real ones on these
collections (0.157 against 0.147, p = 0.15).

## What predicts each component

`mechanism_diagnostics.json` holds label-free statistics of every built index. The
most useful is `value_cv`, the coefficient of variation of the stored document feature
values: saturation compresses large values, so it should matter most where a few
features carry outsized values. `predictor_exploration.json` has the full results.

Saturation's Shapley value, held-out R² (Spearman in brackets):

| Features | Unseen basis family | Unseen corpus | Unseen budgets (322 rows) | Unseen seeds (100 rows) |
|---|---|---|---|---|
| pre-registered three | -0.79 (0.22) | 0.14 (0.48) | 0.15 (0.51) | 0.05 (0.42) |
| `value_cv` alone | 0.32 (0.60) | 0.35 (0.75) | 0.41 (0.82) | 0.66 (0.93) |
| `value_cv`, `peak_ratio`, `length_log_sd` | 0.41 (0.75) | 0.44 (0.76) | 0.49 (0.82) | 0.69 (0.93) |

Length's and IDF's Shapley values are not predictable from any of these diagnostics on
the family and corpus axes (R² at or below zero).

This part is exploratory: the targets and features were chosen after looking at the
primary rows. The budget and seed columns are out of sample in rows, indexes and
diagnostics, but share collections with the training rows.

## Confirmatory tests

`confirmatory_tests.json`, from `scripts/test_hypotheses.py`, on the parameters family
(k1 in {0.6, 1.2, 2.4}, b in {0, 0.5, 0.75}; 630 rows the exploration did not use).
H2-H6 are Holm-corrected together.

| Hypothesis | Result | Outcome |
|---|---|---|
| H1 collection-statistics share at the default cell in 36-42% | 39.3% | pass, consistency check only |
| H2 saturation larger at k1 0.6 than 2.4 | 0.093 against 0.080, p = 3.6e-5 | pass |
| H3 within-corpus Spearman of `value_cv` with saturation at least 0.3 | 0.89 at k1 0.6, 0.89 at k1 2.4 | pass |
| H4 the primary-fitted `value_cv` model ranks saturation with Spearman at least 0.5 | 0.87 at k1 0.6, 0.90 at k1 2.4 | pass |
| H5 length's Shapley value exactly zero at b = 0 | maximum 0.0 over 210 rows | pass |
| H6 length larger at b 0.75 than 0.5 | 0.045 against 0.029, p = 3.6e-5 | pass |

H1's data overlaps the exploratory data, since the default cell of the parameters
family is the primary row for those collections. H5 checks the implementation, not a
scientific claim. The substantive confirmations are H2, H3, H4 and H6: the switches
behave as their mechanisms predict when their strength is varied, and the value-
dispersion predictor keeps its ranking at compression strengths it was not fitted on.

## Reading

Most of the benefit of the BM25 corrections on these representations comes from
saturation, a per-document transform that needs no collection statistics, and its size
follows the dispersion of the stored feature values. Collection statistics account for
a stable 38-39% of the gain, carry real information, and are not predicted by any cheap
diagnostic tried here. IDF alone, the statistic this line of work centres on, accounts
for about a tenth.

## Reproducing

```sh
python -m scripts.shapley_attribution results/setting-a-2026-09-29/analysis/rows.json results/setting-a-2026-09-29/exploratory
python -m scripts.mechanism_diagnostics OUT.json --config configs/workstation-no-msmarco.yaml --output-root OUTPUT_ROOT
python -m scripts.saturation_predictor ROWS DIAGNOSTICS OUT.json
python -m scripts.test_hypotheses ROWS DIAGNOSTICS OUT.json
```

The diagnostics need the built indexes; everything else runs from the committed files.
