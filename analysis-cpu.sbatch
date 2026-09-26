#!/bin/bash
#SBATCH --job-name=csx-analysis
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --nodelist=gnode069
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Missing project directory}}"
source "${VENV_DIR:-.venv}/bin/activate"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/ssd_scratch/$USER/output}"
export CSX_ANALYSIS_WORKERS="${CSX_ANALYSIS_WORKERS:-${SLURM_CPUS_PER_TASK:-36}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SYNC_OUTPUT="${SYNC_OUTPUT:-1}"
unset SCHEDULER_RESUME_INVENTORY EVALUATOR_RESUME_VALIDATION CSX_EVALUATION_BACKEND
if [[ -n "${ANALYSIS_UPDATE_BUNDLE:-}" ]]; then
    python -m scripts.apply_analysis_update "$ANALYSIS_UPDATE_BUNDLE" --output-root "$OUTPUT_ROOT"
fi
# Reuse existing signal forwarding, stage-status recording and final output sync.
exec bash train-cpu.sbatch setting-a analysis fit-predictor report
