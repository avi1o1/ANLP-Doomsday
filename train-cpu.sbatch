#!/bin/bash
#SBATCH --job-name=csx-cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --nodelist=gnode069
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Missing project directory}}"
source "${VENV_DIR:-.venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-/ssd_scratch/$USER/ir-datasets}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/huggingface}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/ssd_scratch/$USER/output}"
mkdir -p "/ssd_scratch/$USER" "$IR_DATASETS_HOME" "$HF_HOME" "$OUTPUT_ROOT"

family="${1:-setting-a}"
shift || true
if [[ "$#" -eq 0 ]]; then
    echo "Usage: sbatch train-cpu.sbatch FAMILY STAGE [STAGE ...]" >&2
    exit 2
fi
echo "CPU stages on $(hostname): $*"
python -m src --config "${CONFIG:-configs/research.yaml}" --output-root "$OUTPUT_ROOT" \
    launch --family "$family" --gpus 0 --workers 1 --cpus "$OMP_NUM_THREADS" --stages "$@"

if [[ "${SYNC_OUTPUT:-0}" == 1 ]]; then
    rsync -a --partial -e 'ssh -o BatchMode=yes' -- "$OUTPUT_ROOT" "ada:/share1/$USER/"
fi
