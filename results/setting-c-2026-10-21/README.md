# Setting C results, expert routing, 21 October 2026 (partial)

The retrieval correction applied to the router of OLMoE-1B-7B (`allenai/OLMoE-1B-7B-0924`,
top-8 routing), on WikiText-2 and zero-shot MMLU. Run on the Ada cluster, jobs 28260 and
30650 (gnode057, 2 GPUs each; the second, submitted by Mohit as `route-rev`, appears from its name to work from the other end of the subject list, which was not checked. The completed subjects cover both ends).
Execution record: `context/cluster-run-2026-10-15/README.md` in the context repository.

**Partial.** WikiText and **52 of 57 MMLU subjects** were complete when these files were pulled.
`transfer_summary.json` was written by the pipeline at that point and will change as the last
subjects finish.

## Contents

| Path | What it is |
|---|---|
| `setting_c_rows.csv` | One line per row: score for the native router, the frequency-bias analogue (the loss-free-balancing comparison), and the eight switch configurations, with margin and p-value. WikiText scores are mean negative log-likelihood per token, MMLU scores are accuracy. |
| `rows/wikitext/`, `rows/mmlu__<subject>/` | `result.json`, `outcomes.npy`, `manifest.json`, `diagnostics.json`, `example_weights.json`, `measurements.json` (mean score per configuration, mean seconds per example). |
| `transfer_summary.json` | The pipeline's scoring of the frozen retrieval predictor against these rows. |

Per-example files are on gnode057 scratch only.

## WikiText-2 (141 windows)

| Router | Mean NLL per token | Perplexity |
|---|---:|---:|
| Native | 2.0134 | 7.49 |
| All switches off | 2.0134 | 7.49 |
| Frequency-bias analogue | 2.0457 | 7.74 |
| All switches on | 2.0649 | 7.88 |

All off matches native, which is the no-op parity check. Margin -2.07, p = 5e-4.

## MMLU (52 subjects, accuracy)

| Router | Unweighted mean over subjects | Weighted by question count |
|---|---:|---:|
| Native (= all off) | 0.480 | 0.484 |
| Frequency-bias analogue | 0.427 | 0.435 |
| All switches on | 0.380 | 0.394 |

All on is below all off in 47 of 52 subjects, above in 3, equal in 2.

## Transfer scoring of the frozen predictor

`transfer_summary.json`, 53 rows (WikiText plus 52 subjects): sign accuracy 0.06, AUC 0.40, MAE
3.96 against the threshold 1.81, so the point thresholds fail. As in Setting B the predictions
are nearly constant, +2.05 to +2.19 (mean +2.13), while the observed margins have mean -1.82
and 3 of 53 are positive; the correlation between predicted and observed is -0.22.
