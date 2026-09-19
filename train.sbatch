#!/bin/bash
#SBATCH --job-name=collectionstats
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --nodelist=gnode083
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail

# Submit from the checkout, or export PROJECT_DIR before calling sbatch.
: "${SLURM_JOB_ID:?Submit this script with sbatch on the cluster}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Missing project directory}}"
source "${VENV_DIR:-.venv}/bin/activate"

export NGPU="${NGPU:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-36}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-/ssd_scratch/$USER/ir-datasets}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/huggingface}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/ssd_scratch/$USER/output}"
# Create the user directory and all required subdirectories before starting training.
mkdir -p "/ssd_scratch/$USER" "$IR_DATASETS_HOME" "$HF_HOME" "$OUTPUT_ROOT"

echo "Starting job on $(hostname)"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-$NGPU}"
echo "Allocated CPUs: $OMP_NUM_THREADS"
echo "Project: $PWD; configuration: ${CONFIG:-configs/research.yaml}; family: ${1:-primary}"
echo "Data: $IR_DATASETS_HOME; models: $HF_HOME; outputs: $OUTPUT_ROOT"

# The runner divides GPU tasks across the allocation and runs CPU stages serially.
# Forwarded USR1/TERM signals preserve completed shards and SAE checkpoints.
csx_args=(--config "${CONFIG:-configs/research.yaml}" --output-root "$OUTPUT_ROOT")
# Keep the shell alive for the final copy while forwarding checkpoint signals.
training_pid=""
wait_interrupted=0
request_stop() {
    wait_interrupted=1
    if [[ -n "$training_pid" ]]; then
        kill -USR1 "$training_pid" 2>/dev/null || true
    fi
}
trap request_stop USR1 TERM INT
python -m src "${csx_args[@]}" launch \
    --family "${1:-primary}" --gpus "$NGPU" \
    --workers "${WORKERS:-4}" --cpus "$OMP_NUM_THREADS" &
training_pid=$!
while :; do
    wait_interrupted=0
    training_status=0
    wait "$training_pid" || training_status=$?
    [[ "$wait_interrupted" -eq 0 ]] && break
done
trap - USR1 TERM INT

# Copy the output directory, preserving local files and existing unrelated remote files.
echo "Syncing $OUTPUT_ROOT to ada:/share1/$USER/"
sync_status=0
rsync -a --partial -e 'ssh -o BatchMode=yes' -- "$OUTPUT_ROOT" "ada:/share1/$USER/" || sync_status=$?
if [[ "$sync_status" -ne 0 ]]; then
    echo "Output sync failed (exit $sync_status); local files remain at $OUTPUT_ROOT" >&2
fi
if [[ "$training_status" -ne 0 ]]; then
    exit "$training_status"
fi
exit "$sync_status"
