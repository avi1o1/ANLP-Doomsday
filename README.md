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
(or `VENV_DIR`), request 36 CPUs, and set the same storage defaults. GPU jobs
request four GPUs and run up to four independent tasks (nine CPU threads each):

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
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --extra ml --extra test --extra benchmarks --frozen
```

The wheel is for Linux x86-64, Python 3.10+ and CUDA 12. The `ml` extra pins a
Torch 2.8 / CUDA 12.8 runtime and GPU FAISS into one environment.

Outputs default to `/ssd_scratch/$USER/output`. The script creates `/ssd_scratch/$USER`
and the cache/output directories before training. After the runner exits, it uses
`rsync` to copy this directory to `ada:/share1/$USER/`, resulting in
`/share1/$USER/output` on `ada`. Local files remain available. This requires
noninteractive SSH access from the compute node and `rsync` on both machines.
Set `OUTPUT_ROOT` for another output directory; it is copied under its own name.
Set `PROJECT_DIR` if submitting from elsewhere. `WORKERS` defaults to four for the four-GPU allocation.
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

To resume after a reviewed code/environment upgrade, run the audit on the compute
node with the existing outputs mounted, then submit from the checkout:

```sh
export OUTPUT_ROOT="/ssd_scratch/$USER/output/research-v5"
python -m scripts.audit_resume "$OUTPUT_ROOT"
export CSX_REUSE_COMPLETED="$OUTPUT_ROOT/resume_approval.json"
bash submit-setting-a.sh --resume
```

This mode assumes preparation and all BM25 baselines already completed. It retains
completed exact CPU dense results as CPU measurements, verifies checksums and query
coverage, and uses GPU FAISS for remaining dense evaluations. Original manifests
and provenance remain unchanged. Incomplete evaluations are archived before restart;
partial representations require separate review. Do not edit code or update the
environment while these jobs are running. CPU stages request no GPUs.

The corpus encoder now supports shared GPU execution through `shared_encoding` in
the experiment config. `launch --stages encode-corpus` partitions each unique
document corpus across four workers, runs E5 once per unfinished batch, and applies
all required dictionaries on the same GPU activations. Token transforms are batched
in bounded chunks; exact feature-value ties prefer lower feature IDs. The final CPU
job runs `join-encoded` before indexing. Shared MS MARCO/TREC-DL documents are encoded
once per representation, while query sets stay separate.

For the reviewed migration of an existing encoding run, stop its launcher with
SIGUSR1, wait for every worker to exit, and run `python -m scripts.audit_resume
"$OUTPUT_ROOT" --encoding` on the compute node. Use a frozen configuration containing
`shared_encoding: {enabled: true, backend: cuda, batch_size: 8,
token_chunk_size: 1024, shards_per_corpus: 4}`. Then submit:

```sh
export CSX_REUSE_COMPLETED="$OUTPUT_ROOT/encoding_resume_approval.json"
bash submit-setting-a.sh --resume-encoding
```

This resumes encoding and queues CPU joining/evaluation without refitting any basis.
Completed artifacts retain original provenance; incomplete shards are archived and
recomputed. CPU and CUDA arithmetic are validated numerically, not claimed bitwise
identical. Batch size 8 is retained for compatibility with existing FP16 E5 outputs;
larger tested batches changed quantized assignments beyond the migration tolerance.
Benchmark tools and tests are in `scripts/benchmark_shared.py` and
`tests/test_shared_encoding.py`.

CPU retrieval evaluation runs up to one independent experiment per allocated CPU
(`CPU_WORKERS` defaults to the allocation CPU count). On a 36-CPU allocation this
allows 36 single-threaded workers while enough unfinished experiments remain. Other CPU stages
remain serial; stage dependencies and artifact locks are retained. Use
`CPU_WORKERS=1` for isolated latency measurements: concurrent timings include
resource contention and must not be compared directly with earlier serial timings.

For an existing run, keep `OUTPUT_ROOT` and the scientific configuration unchanged.
After stopping the old job cooperatively, verify completed artifacts with
`python -m scripts.audit_resume "$OUTPUT_ROOT" --evaluation`, export the resulting
`evaluation_resume_approval.json` as `CSX_REUSE_COMPLETED`, and submit only the
remaining stages:

```bash
export CPU_WORKERS=36
sbatch train-cpu.sbatch setting-a evaluate analysis fit-predictor report
```

Partial checkpoints across a launcher-only update require an explicit source
inventory captured before deployment. `SCHEDULER_RESUME_INVENTORY` enables the
validated migration in `scripts/resume_scheduler.py` before the completion audit.
It rejects changes to scientific source or package versions, validates saved query
batches and parent identities, and preserves the old manifest and batch hashes in
`scheduler_migration.json`. `RESUME_BATCH_SIZE` must match the original evaluation
batch size (default 32). Do not deploy source changes while workers are running.

Transformed postings are cached on local SSD at
`CSX_POSTINGS_CACHE=/ssd_scratch/$USER/postings-cache`. The global cache is bounded
by `CSX_POSTINGS_CACHE_GIB=192`, shared between workers, and evicts least-recently
used entries. It leaves at least 32 GiB free before writing new entries and falls
back to uncached scoring if space is scarce. Keys include parent artifact identity,
statistics, all scoring switches/parameters, and implementation hashes. Cache files
use checksummed NPZ arrays; corrupt entries rebuild automatically. Research outputs
remain under `OUTPUT_ROOT`; the disposable cache is not included in the final rsync.
The first query batch builds entries; subsequent batches reuse transformed CSC
postings without decompressing source representations or reapplying corrections.

New query batches record cache use and worker concurrency. Earlier saved batches
retain their original timings. Caching and concurrency change execution costs, so
mixed timings are not an isolated scoring benchmark. For this evaluator update,
`EVALUATOR_RESUME_VALIDATION` points to a production-shard parity certificate made
by `scripts/validate_postings_cache.py` before checkpoint migration. Migration
rejects changes outside the explicitly validated execution files, preserves prior
migration records, and retains completed batches byte-for-byte.

GPU retrieval evaluation is available through `evaluate-gpu.sbatch`:

```bash
# Set CONFIG, OUTPUT_ROOT and VENV_DIR to the existing run/environment first.
sbatch evaluate-gpu.sbatch setting-a evaluate
```

The default request is one GPU and 36 CPUs on gnode069. Each GPU runs one independent
evaluation; request `--gres=gpu:4` and export `NGPU=4 WORKERS=4` when four GPUs are
available. `CSX_EVALUATION_BACKEND=cuda` activates ordered FP32 scoring and GPU
lexicographic top-k merging. `CSX_GPU_QUERY_BATCH=512` groups missing queries for
one corpus pass, while preserving the original 32-query checkpoint boundaries.
Completed CPU batches remain untouched. A stop request discards only the current
unfinished GPU group; already saved groups resume normally.

The GPU path intentionally uses separately rounded multiplication and addition in
feature-ID order rather than GEMM/TF32/mixed precision. The production validation
script `scripts/validate_gpu_scoring.py` checks CPU/GPU rankings and posting visits,
then benchmarks a 512-query group. A source-bound validation certificate is required
for migration across this evaluator change. Existing timings retain their execution
metadata; new batches report `cuda_ordered_fp32` and the actual query group size.
The postings cache remains shared and bounded. Submit analysis/predictor/report as
a dependent CPU job; no GPU is needed for those stages.

CPU analysis supports `CSX_ANALYSIS_WORKERS` (default 1 for interactive calls).
`analysis-cpu.sbatch` sets it to the allocated CPU count: component ablations can
use up to 36 independent processes, and the 12 predictor-validation variants can
run concurrently. Each worker uses one BLAS/OpenMP thread. Seeds, fold boundaries,
regularization selection, bootstrap draws, Holm correction and result ordering
remain unchanged. Reporting remains serial because it mainly writes small tables.

Diagnostics are cached under `analysis/diagnostics/<input-and-code-hash>/`, with
atomic per-representation ablation checkpoints. Predictor fitting reuses that pass
rather than repeating all bootstraps. Changed outcomes, configuration, result rows
or source provenance select a fresh cache directory; canonical tables remain under
`analysis/`. The pending cluster analysis update is installed only after the GPU
prerequisite succeeds, through `scripts/apply_analysis_update.py`. The installer
verifies the staged source inventory and saves before/after provenance under
`analysis/deployment/`; it never modifies the source of a running GPU job.
