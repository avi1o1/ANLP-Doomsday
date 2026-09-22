#!/bin/bash
#SBATCH --job-name=csx-gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --nodelist=gnode069
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Missing project directory}}"
source "${VENV_DIR:-.venv}/bin/activate"
export NGPU="${NGPU:-4}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-36}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-/ssd_scratch/$USER/ir-datasets}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/huggingface}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/ssd_scratch/$USER/output}"
mkdir -p "/ssd_scratch/$USER" "$IR_DATASETS_HOME" "$HF_HOME" "$OUTPUT_ROOT"

family="${1:-setting-a}"
shift || true
if [[ "$#" -eq 0 ]]; then
    echo "Usage: sbatch train-gpu.sbatch FAMILY STAGE [STAGE ...]" >&2
    exit 2
fi
echo "GPU stages on $(hostname): $*; GPUs: ${SLURM_GPUS_ON_NODE:-$NGPU}"
python -m src --config "${CONFIG:-configs/research.yaml}" --output-root "$OUTPUT_ROOT" \
    launch --family "$family" --gpus "$NGPU" --workers "${WORKERS:-4}" \
    --cpus "$OMP_NUM_THREADS" --stages "$@"
