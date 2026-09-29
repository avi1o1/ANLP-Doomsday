# Hypotheses fixed before the confirmatory tests

Written and committed on 29 September 2026, before any test below was run. The
exploratory analysis that suggested them used the primary rows, the sparsity budgets,
the robustness seeds and the controls. None of it used the parameters family, which
sweeps k1 over {0.6, 1.2, 2.4} and b over {0, 0.5, 0.75} on scifact, fiqa and the three
MIRACL collections, 630 rows. That family is the confirmatory data.

Attribution is the Shapley value of each switch over the eight scoring configurations,
computed per row from its nDCG@10 in each configuration. The label-free diagnostic is
value_cv, the coefficient of variation of the stored document feature values in the
row's index.

Each test is two-sided at 0.05 unless stated; the family of H2-H6 is Holm-corrected.

- H1. Attribution replicates. On the parameters rows at the default cell (k1 1.2,
  b 0.75), the collection-statistics share (IDF plus length) of the all-on gain lies in
  the primary run's 95% interval, 36-42%.
- H2. Saturation strengthens as compression strengthens. Saturation's Shapley value is
  larger at k1 = 0.6 than at k1 = 2.4 at b = 0.75, paired by representation and
  collection (Wilcoxon signed-rank, one-sided).
- H3. The mechanism holds at every k1. At k1 = 0.6 and at k1 = 2.4 (b = 0.75), the
  within-corpus Spearman correlation between value_cv and saturation's Shapley value is
  positive and at least 0.3.
- H4. The predictor transfers across k1. A ridge model on value_cv fitted on the
  primary rows (k1 1.2) ranks saturation's Shapley value on the k1 = 0.6 and k1 = 2.4
  rows with Spearman at least 0.5 in each.
- H5. Length normalisation is inert at b = 0. Length's Shapley value is exactly zero,
  to floating-point tolerance, on every b = 0 row. This checks the implementation, not
  the science: at b = 0 the length factor is 1 for every document.
- H6. Length strengthens with b. Length's Shapley value is larger at b = 0.75 than at
  b = 0.5 at k1 = 1.2, paired (Wilcoxon signed-rank, one-sided).

A hypothesis that fails is reported as failing.
