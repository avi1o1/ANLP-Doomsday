#!/bin/bash
#SBATCH --job-name=csx-eval-gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --nodelist=gnode069
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Missing project directory}}"
source "${VENV_DIR:-.venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-36}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-/ssd_scratch/$USER/ir-datasets}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/huggingface}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/ssd_scratch/$USER/output}"
mkdir -p "/ssd_scratch/$USER" "$IR_DATASETS_HOME" "$HF_HOME" "$OUTPUT_ROOT"

export NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-1}}"
export CSX_EVALUATION_BACKEND=cuda
export CSX_GPU_QUERY_BATCH="${CSX_GPU_QUERY_BATCH:-512}"
# Migrate only after a production GPU parity certificate has been generated.
if [[ -n "${SCHEDULER_RESUME_INVENTORY:-}" ]]; then
    migration_args=()
    if [[ -n "${EVALUATOR_RESUME_VALIDATION:-}" ]]; then
        migration_args+=(--evaluator-validation "$EVALUATOR_RESUME_VALIDATION")
    fi
    python -m scripts.resume_scheduler "$OUTPUT_ROOT" "$SCHEDULER_RESUME_INVENTORY" \
        --batch-size "${RESUME_BATCH_SIZE:-32}" "${migration_args[@]}"
    python -m scripts.audit_resume "$OUTPUT_ROOT" --evaluation
    export CSX_REUSE_COMPLETED="$OUTPUT_ROOT/evaluation_resume_approval.json"
fi

export CSX_POSTINGS_CACHE="${CSX_POSTINGS_CACHE:-/ssd_scratch/$USER/postings-cache}"
export CSX_POSTINGS_CACHE_GIB="${CSX_POSTINGS_CACHE_GIB:-192}"

family="${1:-setting-a}"
shift || true
if [[ "$#" -eq 0 ]]; then
    echo "Usage: sbatch evaluate-gpu.sbatch setting-a evaluate" >&2
    exit 2
fi
echo "GPU evaluation on $(hostname): $*; GPUs=$NGPU; query group=$CSX_GPU_QUERY_BATCH"
training_pid=""
wait_interrupted=0
request_stop() {
    wait_interrupted=1
    if [[ -n "$training_pid" ]]; then
        kill -USR1 "$training_pid" 2>/dev/null || true
    fi
}
trap request_stop USR1 TERM INT
python -m src --config "${CONFIG:-configs/research.yaml}" --output-root "$OUTPUT_ROOT" \
    launch --family "$family" --gpus "$NGPU" --workers "${WORKERS:-$NGPU}" --cpus "$OMP_NUM_THREADS" --stages "$@" &
training_pid=$!
while :; do
    wait_interrupted=0
    training_status=0
    wait "$training_pid" || training_status=$?
    [[ "$wait_interrupted" -eq 0 ]] && break
done
trap - USR1 TERM INT
[[ "$training_status" -eq 0 ]] || exit "$training_status"

if [[ "${SYNC_OUTPUT:-0}" == 1 ]]; then
    rsync -a --partial -e 'ssh -o BatchMode=yes' -- "$OUTPUT_ROOT" "ada:/share1/$USER/"
fi
