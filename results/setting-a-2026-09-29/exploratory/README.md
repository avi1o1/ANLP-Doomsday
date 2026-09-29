# Where the gain comes from, and what predicts it

An analysis of the completed Setting A rows beyond the pre-registered margin
regression. Part of it is exploratory and part confirmatory, and each result below says
which. The confirmatory hypotheses were committed in `HYPOTHESES.md` (commit 9df6d58,
29 September 2026 12:55 IST) before any of their tests ran.

## Why the pre-registered target was hard to predict

The primary margin standardises the all-on minus all-off difference by the spread of
quality over all eight configurations. The all-off configuration is a raw inner product
of non-negative features, and in 42% of the 140 primary rows its nDCG@10 is below 0.01.
The margin sits between 1.5 and 3.5 in 83% of the 137 rows where it is defined. It
mostly records that the uncorrected scorer is broken, and leaves little variation for
any diagnostic to explain. All switches on is the best of the eight configurations in
only 27 of 140 rows.

## Attribution of the gain

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
one-sided Wilcoxon p = 8.9e-7) and zero with uniform ones by construction. The
collection's frequencies carry information that a random reweighting does not. The
effect on quality is small, and all-on nDCG@10 with uniform frequencies is not lower
than with real ones on these collections (0.157 against 0.147, p = 0.15).

## What predicts each component

`scripts/predictor_study.py` compares four tiers of label-free features, each adding to
the one before, and writes `predictor_study.json`.

- pre-registered: `frequency_slope`, `idf_variance`, `frequent_activation_share`;
- index: statistics of the stored document vectors (`mechanism_diagnostics.json`), of
  which the most useful is `value_cv`, the coefficient of variation of the stored values;
- rankings: how much each switch changes the top 10 and the shape of the top-100 scores,
  read from the stored rankings (`ranking_features.json`, `scripts/ranking_features.py`);
- pseudo-labels: every target recomputed with the top 10 of text BM25 on the same
  queries standing in for the relevance judgements.

The targets are the three Shapley values and the collection gain, which is nDCG@10 with
all switches on minus nDCG@10 with saturation alone. The collection gain is the value of
IDF and length normalisation once saturation is applied, the quantity the project
question asks about, and it is positive in 67% of primary rows.

The table gives held-out R² (Spearman in brackets) of ridge regression fitted on the 139
primary rows that have every feature. "Family" and "corpus" hold out whole basis families or whole
corpora; the last column fits on primary rows and predicts the 560 rows at other k1 and
b, which share collections with the training rows.

| Target | Tier | Unseen family | Unseen corpus | Other parameters |
|---|---|---|---|---|
| saturation | pre-registered | -0.46 (0.23) | 0.14 (0.45) | 0.02 (0.41) |
| | index | 0.47 (0.79) | 0.42 (0.81) | 0.50 (0.85) |
| | rankings | 0.53 (0.80) | 0.52 (0.79) | 0.49 (0.86) |
| | pseudo-labels | 0.56 (0.88) | 0.62 (0.85) | 0.76 (0.95) |
| length | pre-registered | -0.30 (-0.10) | 0.14 (0.48) | 0.00 (0.32) |
| | index | -0.16 (0.37) | 0.25 (0.68) | -0.02 (0.42) |
| | rankings | -0.06 (0.28) | 0.20 (0.68) | 0.47 (0.71) |
| | pseudo-labels | 0.49 (0.76) | 0.70 (0.83) | 0.77 (0.73) |
| idf | pre-registered | -0.01 (0.16) | 0.00 (0.29) | 0.02 (0.46) |
| | index | 0.02 (0.20) | 0.02 (0.28) | 0.07 (0.44) |
| | rankings | 0.01 (0.23) | 0.03 (0.23) | 0.20 (0.51) |
| | pseudo-labels | 0.40 (0.60) | 0.21 (0.71) | 0.73 (0.82) |
| collection gain | pre-registered | -0.07 (-0.16) | -0.02 (0.17) | 0.16 (0.38) |
| | index | -0.05 (0.00) | -0.09 (0.06) | 0.15 (0.33) |
| | rankings | -0.14 (-0.07) | -0.17 (0.12) | 0.12 (0.32) |
| | pseudo-labels | 0.38 (0.58) | 0.52 (0.63) | 0.62 (0.59) |

Index statistics predict saturation. For length they reach R² 0.25 on unseen corpora and
fail on unseen families, and they do not predict IDF or the collection gain. Only the
pseudo-labels predict length, IDF and the collection gain on both axes. The pseudo-label
targets, used directly with no fitting, already rank the true values well (Spearman on
the primary rows 0.91 saturation, 0.82 length, 0.83 IDF, 0.66 collection gain).
Gradient-boosted trees (also in the JSON) are not better on the family and corpus axes;
on rows that share collections with the training rows they score higher, but so do
boosted trees on the pre-registered features alone (R² 0.77 at other parameters), so
those columns reward memorising collections and are not evidence of transfer.

### Using the prediction to decide

The decision the project is about is whether to use collection statistics for a given
representation and collection. The table compares using them always, never, only where
the unfitted pseudo-label gain is positive, and where the true gain is positive.

| Rows | Always use them | Never | Pseudo-label rule | Oracle | Headroom recovered | Rule minus always, 95% interval |
|---|---:|---:|---:|---:|---:|---|
| primary (139) | 0.1429 | 0.1279 | 0.1478 | 0.1510 | 61% | 0.0003 to 0.0101 |
| other budgets (320) | 0.1481 | 0.1219 | 0.1537 | 0.1564 | 67% | 0.0003 to 0.0122 |
| other seeds (100) | 0.1556 | 0.1440 | 0.1571 | 0.1624 | 22% | -0.00002 to 0.0045 |
| other parameters (560) | 0.1702 | 0.1535 | 0.1755 | 0.1783 | 65% | 0.00004 to 0.0153 |

Values are mean nDCG@10; intervals resample whole collections. Fitted models do not beat
the unfitted rule. The pseudo-label ridge recovers 58% (unseen family) and 47% (unseen
corpus), and the index-only ridge recovers 11% and -47%. The rule has no parameters,
so its primary-row result is not an in-sample fit. The rule itself was chosen during the
exploration.

By collection, the rule's gain comes from SciFact (0.012 against -0.012 for always using
them) and NFCorpus (0.012 against 0.000). It is at chance on LIMIT and TREC-COVID, where
the true gain averages 0.002 and -0.004, so wrong decisions there cost little. Its
agreement with the true sign does not follow BM25's own quality. BM25 reaches nDCG@10
0.94 on LIMIT and 0.15 on SCIDOCS, where the rule agrees with the true sign in 55% and
98% of rows.

### Choice of reference

The tables above use text BM25 as the reference. `predictor_study_dense.json` and
`predictor_study_fused.json` repeat the study with the dense encoder, and with BM25 and
the dense encoder fused by reciprocal rank.

| Reference | Within-row agreement | Collection gain R², unseen family / corpus | Rule headroom recovered, primary / budgets / seeds / parameters |
|---|---:|---|---|
| text BM25 | 0.98 | 0.38 / 0.52 | 61% / 67% / 22% / 65% |
| dense encoder | 0.98 | 0.51 / 0.66 | 45% / 65% / 49% / 75% |
| fused | 0.98 | 0.50 / 0.61 | 53% / 70% / 23% / 64% |

Within-row agreement is the median Spearman correlation, over the eight configurations
of a row, between true and pseudo-label nDCG@10 (1112 rows with variation in both). The
rule's interval excludes zero in every column except seeds for BM25 and the fused
reference, and primary for the dense encoder. The two references carry different
biases. The sparse representations are built from the same multilingual E5 encoder as the
dense reference, so agreement with it partly measures how much of the source embedding
a configuration keeps. BM25 is independent of the encoder, but lexical systems
contributed to the pools from which several collections' judgements were drawn. The prediction holds
with either reference, so it does not rest on one of these biases. Agreement is lowest
on LIMIT (0.65-0.69) and TREC-COVID (0.74-0.81) for every reference.

This section is exploratory. The features were chosen after the confirmatory tests were
fixed, and none of it was pre-registered.

## Why the components behave as they do

**Saturation.** With length normalisation off, saturation replaces a stored value f by
g(f) = (k1 + 1) f / (f + k1). The factor g(f) / f = (k1 + 1) / (f + k1) falls as f
grows, so the transform shrinks large values relative to small ones. If every stored
value were the same, g would rescale all scores by one factor and leave every ranking
unchanged. The change it makes grows with how unequal the values are, which is what
`value_cv` measures. Smaller k1 bends g more, so saturation should matter more at
k1 = 0.6 than at 2.4 (H2, confirmed). The direction of the effect is almost always
helpful. All-off, a raw inner product of non-negative features, has nDCG@10 below 0.01
in 42% of primary rows, consistent with a few very large coordinates dominating the
raw scores, which saturation caps. The index alone predicts saturation
because whether and how much it helps is a property of the stored values.

**Length normalisation and IDF.** Length normalisation divides by
1 - b + b · |d| / avgdl and IDF weights a coordinate by the rarity of its non-zero
entries. Each changes the ranking by an amount fixed by the index, which the disruption
features measure. Whether the change helps depends on whether heavy documents are
over-rewarded relative to their relevance, and on whether rare coordinates separate
relevant from non-relevant documents. Both are properties of the index jointly
with relevance, so no statistic of the index alone can fix their sign, and none of the
index or ranking features predicts them on unseen families. b = 0 removes
length normalisation exactly (H5), and its contribution grows with b (H6).

**Why pseudo-labels work.** The Shapley values and the collection gain are fixed linear
combinations of the eight nDCG@10 values of a row. If a reference retriever's top 10
agrees with the relevance judgements better than chance, nDCG against that top 10 moves
with the true nDCG from configuration to configuration, and so do the linear
combinations. The measured agreement is close to this ideal (median within-row Spearman
0.98). Text BM25 is built from the collection's words and does not depend on the
representation under test, so it supplies the missing relevance information without
labels. Its cost is one lexical retrieval run over the queries.

**Limitations.** Each reference carries a bias (see the choice of reference), and the
result holds with both. The rule needs queries, and so cannot be applied
before any query is seen. All of it is on ten collections from one encoder.

## Earlier exploration

`predictor_exploration.json` (`scripts/saturation_predictor.py`) is the first pass,
which fitted index statistics to each component and led to the confirmatory tests
below.

| Features | Unseen basis family | Unseen corpus | Unseen budgets (322 rows) | Unseen seeds (100 rows) |
|---|---|---|---|---|
| pre-registered three | -0.79 (0.22) | 0.14 (0.48) | 0.15 (0.51) | 0.05 (0.42) |
| `value_cv` alone | 0.32 (0.60) | 0.35 (0.75) | 0.41 (0.82) | 0.66 (0.93) |
| `value_cv`, `peak_ratio`, `length_log_sd` | 0.41 (0.75) | 0.44 (0.76) | 0.49 (0.82) | 0.69 (0.93) |

These are R² (Spearman) for saturation's Shapley value. The targets and features were
chosen after looking at the primary rows. The budget and seed columns are out of sample
in rows, indexes and diagnostics, but share collections with the training rows.

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
family is the primary row for those collections. H5 is a check of the implementation.
The substantive confirmations are H2, H3, H4 and H6. Under them the switches behave as
their mechanisms predict when their strength is varied, and the value-dispersion
predictor keeps its ranking at compression strengths it was not fitted on.

## Reading

Most of the benefit of the BM25 corrections on these representations comes from
saturation, a per-document transform that needs no collection statistics. Its size
follows the dispersion of the stored values, as the mechanism predicts, and the index
alone predicts it. Collection statistics account for a stable 38-39% of the gain and
carry real information, but whether they help a given row is not a property of the
index, since it depends on relevance. A label-free proxy for relevance, agreement with
text BM25 or with the dense encoder, predicts it. Choosing by the BM25 proxy recovers
61-67% of the gap between always using collection statistics and an oracle on the
primary rows and at other budgets and parameters, and 22% at other seeds, where the
interval reaches zero. IDF alone, the statistic this line of work centres on, accounts
for about a tenth of the gain.

## Reproducing

```sh
python -m scripts.shapley_attribution results/setting-a-2026-09-29/analysis/rows.json results/setting-a-2026-09-29/exploratory
python -m scripts.mechanism_diagnostics OUT.json --config configs/workstation-no-msmarco.yaml --output-root OUTPUT_ROOT
python -m scripts.ranking_features OUTPUT_ROOT RANKING_DIR
python -m scripts.saturation_predictor ROWS DIAGNOSTICS OUT.json
python -m scripts.test_hypotheses ROWS DIAGNOSTICS OUT.json
python -m scripts.predictor_study ROWS DIAGNOSTICS RANKING_DIR.json OUT.json [--reference dense|fused]
```

The diagnostics and ranking features need the built indexes and stored evaluation
records; everything else runs from the committed files.
