# Transfer registration, 20 October 2026

This file fixes, before the corrected outcomes of Settings B and C are read, which
predictors are tested, on which rows, against which baselines and with which pass marks.
It also fixes one further retrieval test on SPLADE-v3. Anything analysed that is not
listed here is exploratory and is reported as such.

## What had been seen when this was written

- **Setting B (attention)** ran as cluster job 28259 from 19 October 15:02 to 20 October
  16:31 IST. All 26 rows are complete, each with 100 examples and their stored calibration
  references and full-attention outputs.
  - `src/transfer.py` writes `transfer/attention/summary.json` after every completed row.
    It scores the frozen predictor of Section 1 against that row's margin.
  - Neither that file nor any row's `result.json`, `outcomes.npy` or per-configuration
    output has been opened (Anirudh, 20 October).
  - Before this commit we checked only file counts, manifests, the predictor hash and the
    data hashes.
- **Setting C (routing)** is running as job 28260. No outcome has been read.
- SPLADE-v3 outcomes under all eight configurations are known
  (`results/setting-a-2026-10-18/splade/`). The label-free estimates of Section 4 had not
  been computed.
- The unfitted decision rule of the mid report had been evaluated on the 84 MS MARCO
  budget rows (Section 5). Those results are stated there and are not a registered test.

## 1. The pre-registered diagnostic predictor (Settings B and C)

- **Model.** `results/setting-a-2026-09-26/research-v5/predictor/predictor.json`,
  sha256 `ff95b4f61f8702b1ce9c117b91d181302860fe44493a2da8feb9c2ee9c038fc2`.
  - It is a ridge regression of the margin m on `frequency_slope`, `idf_variance` and
    `frequent_activation_share`, fitted on 179 of the 182 primary retrieval rows and
    committed on 26 September. The other three have zero spread across configurations,
    so their margin is undefined.
  - No refit, no recalibration and no change of features. A transfer `summary.json`
    whose `predictor_sha256` differs from this one is not a result of this test.
- **Target.** The margin m = (Q111 − Q000)/s of each transfer row, as computed by
  `paired_effect`.
- **Pass marks (unchanged since 19 September, `src/analysis.py`).**
  - Sign accuracy ≥ 0.70.
  - Mean absolute error ≤ 1.5 × 1.209 = 1.81.
- **Baseline.** The constant prediction of the retrieval majority sign, m > 0, which held
  on 89.9% of the training rows. Every result is reported against it.
  - A predictor that does not beat it has not shown transfer, whatever the pass marks
    say.
  - The better of the two constant signs on the transfer rows is also reported, as the
    strictest comparison.
- **Rows with undefined margins** (all eight configurations equal) are excluded and
  counted.
- **Uncertainty.** A bootstrap over tasks (Setting B) or over MMLU subjects (Setting C),
  holding the predictor fixed.

## 2. The label-free decision rule in Setting B

This carries the mid report's pseudo-label predictor into attention with a native
reference. There are no fitted parameters.

- **Primary reference: exact attention mass, as stated in the mid report.**
  - For each example, the uncorrected model generates greedily with full attention. At
    every decode step, in every layer and head, we record the fraction of the exact
    softmax attention mass that falls on the tokens each configuration would have
    selected.
  - The selection uses that configuration's scoring, the same budget, and the frozen
    calibration reference of that example.
  - The pseudo-quality q̃_c of a row is that fraction averaged over steps, layers, heads
    and examples.
  - Code: `scripts/attention_mass_reference.py`. It reads only the uncorrected
    generation and the calibration reference, never a corrected outcome.
- **Estimate.** G̃ = q̃111 − q̃010.
- **Rule.** Use collection statistics (configuration 111 rather than 010) where G̃ > 0.
- **Target.** G = Q111 − Q010 in RULER accuracy.
- **Reported.**
  - Sign accuracy of G̃ against the sign of G, set beside two constant predictions:
    G > 0 (the retrieval majority, 71.4% of the primary rows) and the better of the two
    constant signs on the transfer rows themselves.
  - Spearman correlation between G̃ and G.
  - The rule's mean accuracy against always-on (111), always-off (010) and the per-row
    oracle, with the share of the headroom recovered and a task bootstrap interval.
- **Pass.** Both are needed:
  - the sign accuracy exceeds the better constant sign on the transfer rows;
  - the rule's mean accuracy is at least always-on's.
- **Secondary reference.** Agreement of each configuration's output with the
  full-attention output (token F1 after RULER's cleaning), which needs no new run.
  - It is reported beside the primary and does not replace it.
  - If the primary reference cannot be computed for all 26 rows by 26 October, this
    registration is amended in a dated commit before any G is read. The secondary then
    becomes the test, and the change is stated in the report.

## 3. Setting C (routing)

- **Rows.** WikiText-2 (one row) and every MMLU subject (57 rows).
  - MMLU is not cut. Without it the test of Section 1 has a single row.
  - If time requires, MMLU is subsampled to the first N questions of every subject in
    stored order, with N fixed before any outcome is read and the same for all subjects.
- **Tested.** The predictor of Section 1.
- **Label-free reference.** The dense softmax mixture named in the mid report is not
  implemented, so no label-free rule is tested in routing; the report says so.
  WikiText negative log-likelihood is itself label-free, so the rule would add nothing
  there.

## 4. SPLADE-v3: does the label-free rule detect the reversal?

- **Rows.** The 10 collections outside MIRACL. On MIRACL every SPLADE-v3 score is near
  zero.
- **Needs** SPLADE-v3's stored rankings under each of the eight configurations. If they
  were not kept, they are regenerated from the stored features before any G̃ is computed.
- **Estimate.**
  - G̃ from agreement with a reference's top 10, as in the mid report, on SPLADE-v3's
    rankings under each configuration.
  - Primary reference: text BM25, the reference of the mid report's rule.
  - Dense E5 and the fused reference are secondary.
- **Rule, target and reported quantities** as in Section 2, with nDCG@10 in place of
  RULER accuracy.
- **Baseline.**
  - The baseline is always-on (the mid report's default), plus the constant sign G > 0.
  - Here G is negative on 5 of the 10 collections, so the rule has a real choice to make.
- **Known risk.** Agreement with BM25 may favour the BM25-like correction, the bias the
  mid report names. The dense reference is reported for that reason.

## 5. Already evaluated, not registered

The mid report's rule (BM25 reference, G̃ > 0) on the 84 MS MARCO budget rows produced
after it was published:

| Reference | Sign accuracy | Constant baseline | Headroom recovered |
|---|---:|---:|---:|
| BM25 | 0.869 | 0.881 | 21% |
| Dense E5 | 0.952 | 0.881 | 77% |
| Fused | 0.952 | 0.881 | 66% |

- Always-on scores 0.2021 and the oracle 0.2028.
- The three collections share one corpus.

## 6. Scope of Setting B

- **Model.** Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  in float16. This is the fallback named in the configuration (`attention.fallback_model`),
  used because Llama-3.2-1B-Instruct was not accessible on the cluster.
- **Data.** All 13 RULER tasks at 4K context, RULER revision
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`, seed 0, 100 examples per task. Each row
  records the sha256 of its data.
- **Budgets.** 512 and 1,024 tokens, with page size 16, giving 26 rows.
- **Modes.** Full attention, Quest, H2O and all eight configurations.

Two deviations from the mid report's plan, both for compute:

- **The 8K context is dropped.** This is the first cut the mid report named.
- **The number of examples falls from 500 to 100.**

The 100-example pass took 25.5 hours on two GPUs. A second, disjoint set of 100 examples
per task may be added to all 26 rows before 25 October, generated with RULER seed 1. The decision depends on GPU time alone. The
analysis uses every completed set, identically for every row. No row is extended on its
own.

## 7. Descriptive analyses (no pass mark)

- Per-row comparison of 111 against 000, with Holm correction over the rows of each
  setting.
- Shapley shares of IDF, saturation and length in Settings B and C, set beside the
  retrieval shares.
- Configuration 111 against Quest (000), H2O and full attention.
