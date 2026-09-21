# Predicting When Collection Statistics Help Top-k Selection

Advanced NLP, Monsoon 2026. Team ANLP Doomsday: Aviral Gupta, Mohit Kumar Singh,
Anirudh Sankar, Arihant Tripathy.

Learned sparse retrieval, block-sparse attention over KV pages, and mixture-of-experts
routing all score a query against a set of learned non-negative features by inner product
and keep the top $k$. None of them uses how often a feature is selected across the
collection, which is the statistic BM25 is built on. We build eight concept bases from one
frozen encoder, score each with a function whose IDF, saturation, length-normalisation and
top-$k$ components switch independently, and regress the resulting change in quality on six
cheap measurements of the basis. The fitted predictor is then applied, without refitting, to
KV-page selection and to expert routing, and to Hindi, Bengali, Telugu and Hindi-English
code-mixed retrieval, where the distributions it depends on are known to shift.

## Layout

| Path        | Contents                                                   |
| ----------- | ---------------------------------------------------------- |
| `docs/`     | LaTeX sources for all four written deliverables, ACL style |
| `docs/pdf/` | built PDFs under their submission names                    |
| `slides/`   | viva presentation                                          |
| `Makefile`  | entry point for every command; `make` lists them           |
| `make/`     | shared Makefile pieces, included by the rest               |

## Deliverables

| Deliverable      | Due         | File                        |
| ---------------- | ----------- | --------------------------- |
| Interim proposal | 14 Aug 2026 | `ANLPDoomsday-Interim.pdf`  |
| Final proposal   | 28 Aug 2026 | `ANLPDoomsday-Proposal.pdf` |
| Mid submission   | 30 Sep 2026 | `ANLPDoomsday-Mid.zip`      |
| Final submission | 31 Oct 2026 | `ANLPDoomsday-Final.zip`    |

Each member submits every deliverable separately.

## Build

A bare `make` lists every target with a description, so it is the place to look rather
than this file.

```sh
make            # list the targets
make docs       # build all four documents
make proposal   # build one document
make check      # log, page limits, bibliography consistency
make mid-zip    # package the mid submission
make final-zip  # package the final submission
```

New targets are documented by writing `## description` on the target line, and grouped
with a `##@ Group` line. Set `NO_COLOR=1` for plain output.

Needs TeX Live with `natbib`, `times`, `microtype` and `booktabs`. The ACL style files are
bundled in `docs/`, so nothing external is fetched.

## Models and runs

Links to any uploaded model or logged run go here.

## Implementation

The original proposal and deliverable record above is retained. The current
implementation follows the protocol and execution workflow below.

This repository implements controlled retrieval experiments with one frozen multilingual
E5 encoder, seven concept bases at pooled/token granularity and eight independent
IDF/saturation/length scoring configurations. A grouped ridge predictor estimates the
all-on versus all-off margin and transfers without refitting to attention page selection
and OLMoE routing.

Multilingual evaluation uses **Hindi, Bengali and Telugu MIRACL**, within language.
English controls, MS MARCO/TREC-DL and LIMIT cover other collection regimes. There are
no code-mixed experiments, deadline-driven schedules or required row counts. E5 has
MIRACL training exposure; claims concern fitted bases, scoring and prediction, not
encoder generalization to unseen languages.

### Start locally

Use Python 3.11 or 3.12. Full data/model workloads belong on the Slurm cluster.

```sh
# Create the environment only if it does not already exist.
test -d .venv || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[ml,test,benchmarks]'
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
.venv/bin/csx --config configs/fixture.yaml run-retrieval
.venv/bin/csx --config configs/fixture.yaml report
```

The fixture has 30 invented engineering documents and a deterministic encoder. Its
results are implementation checks, never research evidence. Choose a new output root
after changing code, configuration, dependency versions or parent artifacts.
Completed artifact checksums are verified before reuse.

### Cluster execution

Setting A uses separate [CPU](train-cpu.sbatch) and [GPU](train-gpu.sbatch) jobs.
[submit-setting-a.sh](submit-setting-a.sh) submits their dependency graph, overlapping
the independent BM25S CPU job with dense-baseline and sample-encoding GPU work. Both
job types currently target `gnode069` to preserve node-local SSD data, activate `.venv`
(or `VENV_DIR`), request eight CPUs, and set the same storage defaults. GPU jobs
request one GPU and run one GPU task at a time:

```sh
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-/ssd_scratch/$USER/ir-datasets}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/huggingface}"
```

Use those same exports for interactive tests. See [the cluster guide](context/CLUSTER.md)
for the interactive GPU fixture and environment setup. After freezing model versions:

```sh
export CONFIG=configs/frozen.yaml
export OUTPUT_ROOT="/ssd_scratch/$USER/output/research-v4"
bash submit-setting-a.sh
```

BM25 uses BM25S 0.3.11 with its Numba retrieval backend and mmap-loaded CSC index.
It remains a CPU job. Dense encoding, SAE fitting and corpus encoding use GPU jobs;
indexing, evaluation, diagnostics, predictor fitting and reporting use CPU jobs.
You can also submit one stage directly, for example
`sbatch train-cpu.sbatch setting-a baselines-cpu`.

Dense reference evaluation uses GPU FAISS shard indexes with 256-query batches. The
full MS MARCO corpus cannot fit in one 11 GB 2080 Ti, so each stored 4,096-vector shard
is searched exactly on the GPU. Install the pinned PyPI GPU wheel into the cluster
environment before submitting a GPU baseline job:

```sh
uv pip uninstall --python "$VENV_DIR/bin/python" faiss-cpu
uv pip install --python "$VENV_DIR/bin/python" --no-deps 'faiss-gpu==1.15.1'
```

The wheel is for Linux x86-64, Python 3.10+ and CUDA 12. It is pinned through the
separate `gpu-faiss` extra because it requires CUDA BLAS 12.6+, while Torch 2.6 pins
CUDA BLAS 12.4. The existing Torch CUDA libraries are retained with `--no-deps`.

Outputs default to `/ssd_scratch/$USER/output`. The script creates `/ssd_scratch/$USER`
and the cache/output directories before training. After the runner exits, it uses
`rsync` to copy this directory to `ada:/share1/$USER/`, resulting in
`/share1/$USER/output` on `ada`. Local files remain available. This requires
noninteractive SSH access from the compute node and `rsync` on both machines.
Set `OUTPUT_ROOT` for another output directory; it is copied under its own name.
Set `PROJECT_DIR` if submitting from elsewhere. `WORKERS` defaults to one for the
one-GPU allocation.
Training does not build LaTeX or package submissions.

### Layout

| Path | Purpose |
|---|---|
| `src/retrieval/` | Retrieval datasets, encoders, bases, SAE fitting, indexing and evaluation |
| `src/attention/` | Attention adapter, RULER preparation and attention experiments |
| `src/routing/` | Expert-routing adapter, benchmark preparation and routing experiments |
| `src/*.py` | Shared CLI, launchers, scoring, diagnostics, predictor, artifacts and reporting |
| `configs/research.yaml` | Scientific defaults and dataset/model manifest |
| `configs/fixture.yaml` | Offline engineering fixture |
| `configs/gpu-smoke.yaml` | Engineering fixture with real E5 encoding and CUDA SAE fitting |
| `context/CLUSTER.md` | Interactive commands, environment variables and storage layout |
| `train-cpu.sbatch`, `train-gpu.sbatch` | Resource-specific Slurm launchers |
| `submit-setting-a.sh` | Setting A dependency-aware submission helper |
| `tests/` | Numerical, recovery, pipeline and tiny-model parity checks |
| `context/IMPLEMENTATION.md` | Commands, artifact formats, validation and limitations |
| `context/PROTOCOL.md` | Scientific definitions and transfer decisions |
| `context/THEORY.md` | Bilinear identity, ranking conditions and counterexamples |
| `docs/proposal.tex` | Original research proposal |

Attention is a **PyTorch reference selector** that retains the full KV cache and reports
selected KV volume separately. It does not claim fused-kernel speed or physical cache
savings. The external Latent Terms path records missing matching artifacts and labels
conceptual replications.

No full-corpus research runs or cluster hardware results are bundled. Training and
prediction stages produce checkpoints; reports list incomplete and unrun experiments.

The source tree is also the Python package: `python -m src` runs the same CLI as
`csx`. Reinstall the editable package after updating an older checkout so the `csx`
entry point uses the new layout. Shared modules at the `src/` root coordinate the
setting packages; each setting owns its benchmark adapters and experiment pipeline.
Make targets continue to build documents and archives. Python commands run local
checks and experiment stages, and the resource-specific Slurm scripts launch cluster work.
