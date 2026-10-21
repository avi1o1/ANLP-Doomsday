# Setting B results, KV-page selection, 21 October 2026

Quest-style page selection on RULER, with the retrieval correction (IDF, saturation, length)
applied to page scores. Run on the Ada cluster, job 28259 (gnode057, 2 GPUs, 1 d 1 h 29 min).
Execution record: `context/cluster-run-2026-10-15/README.md` in the context repository.

## Scope

- Model: Qwen2.5-1.5B-Instruct (the plan names Llama-3.2-1B-Instruct, which was not used).
- 13 RULER tasks, **4K context only**, budgets 512 and 1,024 tokens, page size 16, no sink:
  26 rows. **100 examples per task**, not the planned 500, and no 8K contexts. Both are
  reductions from the schedule in `docs/mid.tex`, made because the original design was
  estimated at about 480 GPU-hours.
- Each example is generated under 11 configurations: full attention, Quest, H2O, and the
  eight on/off combinations of the three switches. Quest and all-off are the same
  computation by construction (the pipeline asserts identical text).
- Code: the batched adapter (`src/attention/adapter.py`, commit b2b452f), which replaced a
  per-head implementation that could not finish at this scale. RULER data were generated with
  two patches to the pinned checkout (`context/cluster-run-2026-10-15/patches/`).

## Contents

| Path | What it is |
|---|---|
| `setting_b_rows.csv` | One line per row: mean accuracy for full, Quest, H2O and the eight switch configurations, the margin, p-value, and mean timing and memory. |
| `rows/<task>__c4096__b<budget>__sink0/` | `result.json` (effect, margin, quality of the eight configurations), `outcomes.npy`, `manifest.json`, `measurements.json` (per-configuration mean score, decode, selection and calibration seconds, resident and peak GPU memory). |
| `transfer_summary.json` | The pipeline's scoring of the frozen retrieval predictor against these rows. |

Per-example generations (about 28,000 files) are on gnode057 scratch only.

## Mean accuracy over the 13 tasks

| Budget | Full | Quest (= all off) | H2O | All on | Best of the 8 |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.811 | 0.705 | 0.181 | 0.022 | 0.723 |
| 1,024 | 0.811 | 0.778 | 0.266 | 0.027 | 0.784 |

- All switches on is never better than all off: 0 of 26 rows (2 tie). The margin is negative in
  all 24 rows where it is defined. The two undefined rows are `vt` at both budgets.
- Some partial combinations help slightly (best of eight is above all off by 0.018 at 512 and
  0.006 at 1,024; for example `cwe` at 512, length only: 0.328 against 0.288).
- H2O is below Quest in 21 rows, above it in 4, and equal in 1. This was not examined further.
- Cost per example on this model: Quest decode 4.9 s on average, calibration 1.8 s, peak GPU
  memory at most 4.2 GiB.
- With 100 examples per row the accuracy resolution is coarse, and the bootstrap p-values sit
  at their floor of 5e-4 in 24 of 26 rows.

## Transfer scoring of the frozen predictor

`transfer_summary.json`, 24 scored rows: sign accuracy 0.00, MAE 4.18 against the pre-registered
threshold 1.81, so the point thresholds fail. The predictions are nearly constant: every
row is predicted between +2.13 and +2.21 (mean +2.18), while the observed margins have mean
-2.00 and none is positive; the correlation between predicted and observed is -0.18. That is,
the predictor, fitted on retrieval where the margin is mostly +1.5 to +3.5, says the
correction helps in every row, and here it hurts in every row. This is what the numbers show; it
is not an interpretation of why.
