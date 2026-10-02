# Predicting When Collection Statistics Help Top-k Selection

Advanced NLP, Monsoon 2026. Team ANLP Doomsday: Aviral Gupta, Mohit Kumar Singh,
Anirudh Sankar, Arihant Tripathy.

Learned sparse retrieval, block-sparse attention over KV pages and mixture-of-experts
routing all select items by a top-k over inner products. Only lexical retrieval corrects
that score, through the IDF, term-frequency saturation and length normalisation of BM25.
This project asks when that correction helps learned features, and whether the answer
can be predicted without relevance labels.

- **Setting A, retrieval (complete).** Fourteen sparse representations from one frozen
  `multilingual-e5-base` encoder (seven bases at pooled and token granularity), each
  scored under all eight on/off combinations of IDF, saturation and length on thirteen
  English and Indic collections.
- **Setting B, KV-page selection** and **Setting C, expert routing (implemented, not yet
  run).** The same correction applied to Quest-style page selection and to OLMoE routing.

The mid report is `docs/pdf/ANLPDoomsday-Mid.pdf`.

## Repository layout

| Path                                               | Contents                                                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `src/`                                             | The `csx` package: scoring, diagnostics, predictor, artifacts, reporting                                |
| `src/retrieval/`, `src/attention/`, `src/routing/` | One package per setting                                                                                 |
| `configs/`                                         | `research.yaml` (full experiment), `fixture.yaml` (offline test), `gpu-smoke.yaml`, single-GPU profiles |
| `scripts/`                                         | Analysis, curation and maintenance scripts                                                              |
| `tests/`                                           | Unit, numerical-parity and recovery tests                                                               |
| `results/`                                         | Curated Setting A results, each directory with its own README                                           |
| `docs/`                                            | LaTeX sources of all reports; built PDFs in `docs/pdf/`                                                 |
| `*.sbatch`, `submit-setting-a.sh`                  | Slurm launchers for the cluster                                                                         |

## Install and test

Python 3.11 or 3.12, with [uv](https://docs.astral.sh/uv/).

```sh
uv sync --extra ml --extra test --extra benchmarks
uv run pytest -q
```

The `ml` extra pins Torch 2.8 for CUDA 12.8 and GPU FAISS (Linux x86-64). On a machine
without a GPU, CUDA-only tests are skipped.

A small offline fixture (30 invented documents, deterministic encoder) runs the whole
pipeline in seconds. Its numbers check the code and are not research results.

```sh
uv run csx --config configs/fixture.yaml run-retrieval
uv run csx --config configs/fixture.yaml report
```

## Reproducing the mid-report analysis

The tables in the mid report come from the files under `results/`. The analysis scripts
read those files directly:

```sh
R=results/setting-a-2026-09-29
uv run python -m scripts.shapley_attribution $R/analysis/rows.json $R/exploratory
uv run python -m scripts.test_hypotheses $R/analysis/rows.json \
    $R/exploratory/mechanism_diagnostics.json $R/exploratory/confirmatory_tests.json
uv run python -m scripts.predictor_study $R/analysis/rows.json \
    $R/exploratory/mechanism_diagnostics.json $R/exploratory/ranking_features.json \
    $R/exploratory/predictor_study.json --reference bm25
```

`results/setting-a-2026-09-29/exploratory/README.md` describes each output. The index
diagnostics and ranking features (`scripts/mechanism_diagnostics.py`,
`scripts/ranking_features.py`) need the built indexes and stored rankings, so they are
committed as JSON. Per-query records, encodings and indexes are too large to bundle.

## Running the full experiment

Every stage is a `csx` subcommand reading `configs/research.yaml`: `prepare`,
`encode-sample`, `fit-bases`, `encode-corpus`, `index`, `evaluate`, `baseline`,
`diagnose`, `fit-predictor`, `report`. `csx launch` runs a family of stages inside one
allocation, and every stage resumes from its last checkpoint.

On the cluster, set the output root and submit the dependency graph:

```sh
export CONFIG=configs/research.yaml
export OUTPUT_ROOT=/ssd_scratch/$USER/output/run
bash submit-setting-a.sh
```

GPU stages (encoding, SAE fitting, dense baselines, GPU evaluation) and CPU stages
(BM25, indexing, analysis, reporting) run as separate jobs through `train-gpu.sbatch`,
`train-cpu.sbatch`, `evaluate-gpu.sbatch` and `analysis-cpu.sbatch`. Their settings come
from environment variables with defaults in each script (`CONFIG`, `OUTPUT_ROOT`,
`VENV_DIR`, `NGPU`, `WORKERS`, `HF_HOME`, `IR_DATASETS_HOME`). A Hugging Face token goes
in `.env` (see `.env.example`).

## Documents and submission

```sh
make            # list targets
make mid        # build the mid report
make check      # log, page limit and bibliography checks
make mid-zip    # dist/ANLPDoomsday-Mid.zip: report and code
```

Building needs TeX Live with `natbib`, `times`, `microtype`, `booktabs` and `tikz`; the
ACL style files are in `docs/`.

| Deliverable      | Due         | File                        |
| ---------------- | ----------- | --------------------------- |
| Interim proposal | 14 Aug 2026 | `ANLPDoomsday-Interim.pdf`  |
| Final proposal   | 28 Aug 2026 | `ANLPDoomsday-Proposal.pdf` |
| Mid submission   | 2 Oct 2026  | `ANLPDoomsday-Mid.zip`      |
| Final submission | 31 Oct 2026 | `ANLPDoomsday-Final.zip`    |

## Models and runs

No models were uploaded to Hugging Face or WandB.
