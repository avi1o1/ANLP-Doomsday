#!/bin/bash
#SBATCH --job-name=collectionstats
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$PWD/data/ir-datasets}"
export HF_HOME="${HF_HOME:-$PWD/data/huggingface}"
export IR_DATASETS_TMP="${IR_DATASETS_TMP:-${SLURM_TMPDIR:-/tmp}/collectionstats-$USER-$SLURM_JOB_ID}"
mkdir -p "$IR_DATASETS_HOME" "$HF_HOME" "$IR_DATASETS_TMP"

echo "Starting job on $(hostname)"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-$NGPU}"
echo "Allocated CPUs: $OMP_NUM_THREADS"
echo "Project: $PWD; configuration: ${CONFIG:-configs/research.yaml}; family: ${1:-primary}"

# The runner divides GPU tasks across the allocation and runs CPU stages serially.
# Forwarded USR1/TERM signals preserve completed shards and SAE checkpoints.
csx_args=(--config "${CONFIG:-configs/research.yaml}")
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    csx_args+=(--output-root "$OUTPUT_ROOT")
fi
exec python -m src "${csx_args[@]}" launch \
    --family "${1:-primary}" --gpus "$NGPU" \
    --workers "${WORKERS:-4}" --cpus "$OMP_NUM_THREADS"
