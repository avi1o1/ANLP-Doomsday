# Transfer results and registered tests, 20 October 2026 onwards

These are the results of the tests registered in `results/registration-2026-10-20.md`
(commit cf602ac). That file was committed before any Setting B or C outcome was read. Each
analysis script was committed before it was run on real data.

| Path | What it is | Made by |
|---|---|---|
| `setting-b/rows.json` | All 26 Setting B rows compiled from the cluster: the row summary, and per example the score of every mode, the decode measurements and the token F1 to the full-attention output | `scripts/collect_attention_rows.py` |
| `setting-b/predictor_summary.json` | The cluster's own scoring of the frozen predictor (sha256 `ff95b4f6…`) on Setting B | `src/transfer.py` |
| `setting-b/rows/` | Each row's `result.json` and `manifest.json` as written on the cluster | copy |
| `setting-b/attention_mass/` | The label-free reference: exact attention mass captured by each configuration, per row | `scripts/attention_mass_reference.py` |
| `setting-b/needle_page_rates.json` | Calibration selection rates of the needle, sink and last pages on 100 single-needle examples | `scripts/needle_page_rates.py` |
| `setting_b_study.json` | The registered Setting B analysis (Sections 1, 2 and 7) | `scripts/transfer_study.py` |
| `setting_c_study.json` | The registered Setting C analysis (Sections 1, 3 and 7) | `scripts/routing_study.py` |
| `splade_label_free.json` | Registered Section 4: the label-free rule on SPLADE-v3 | `scripts/splade_label_free.py` |
| `msmarco_rule_test.json` | The mid report's unfitted rule on the 84 MS MARCO budget rows made after it (not registered) | `scripts/msmarco_rule_test.py` |

## Setup

- **Setting B.**
  - Model: Qwen2.5-1.5B-Instruct (revision 989aa79), the configured fallback.
  - Data: RULER revision c3f5e3b at 4K, seed 0, 100 examples per task, all 13 tasks.
  - Budgets of 512 and 1,024 tokens over 16-token pages.
  - Cluster job 28259; the attention-mass probe ran as jobs 30378, 30379, 30425 and 30426.
  - Every probed example regenerated its stored full-attention output exactly.
- **Setting C.**
  - Model: OLMoE-1B-7B.
  - Data: WikiText-2 and the 57 MMLU subjects, one process per subject.
  - Cluster jobs 28260 and 30650 (the second from the other end of the subject list); all
    58 rows complete on 21 October. The study was run on the cluster over the full
    per-example outputs.
- **Setting B on Llama-3.2-1B-Instruct.** Registered replication (addendum of 23 October in
  the registration file), cluster job 36453, running.
- **SPLADE-v3.**
  - The stored rankings under all eight configurations were scored against the BM25,
    dense and fused references of the original cluster run (job 30449).

## Reproducing the analyses from these files

```sh
uv run python -m scripts.transfer_study results/transfer-2026-10-20/setting-b/rows.json \
    results/transfer-2026-10-20/setting-b/attention_mass \
    results/setting-a-2026-09-26/research-v5/predictor/predictor.json \
    results/setting-a-2026-09-26/research-v5/analysis/rows.json /tmp/setting_b_study.json
uv run python -m scripts.msmarco_rule_test results/setting-a-2026-10-18/analysis/msmarco_rows.json \
    results/setting-a-2026-10-18/exploratory/ranking_features.json /tmp/msmarco_rule_test.json
uv run python -m scripts.final_tables /tmp/tables
```

The SPLADE test and the routing study read cluster directories (stored rankings and
per-example routing outputs) that are too large to commit. The full outputs are archived
on the cluster share.
