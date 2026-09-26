#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == --resume-encoding ]]; then
    : "${OUTPUT_ROOT:?Set the existing run directory}"
    : "${CSX_REUSE_COMPLETED:?Set the validated encoding migration audit}"
    encode=$(sbatch --parsable train-gpu.sbatch setting-a encode-corpus)
    evaluate=$(sbatch --parsable --dependency="afterok:$encode" --export="ALL,SYNC_OUTPUT=${SYNC_OUTPUT:-0}" \
        train-cpu.sbatch setting-a join-encoded index evaluate analysis fit-predictor report)
    printf 'encode=%s evaluate=%s\n' "$encode" "$evaluate"
    exit 0
fi

# Prepare once, overlap the independent CPU BM25S and GPU reference work, then
# join both branches before evaluation. --parsable returns only each Slurm job ID.
if [[ "${1:-}" == --resume ]]; then
    : "${OUTPUT_ROOT:?Set the existing run directory}"
    : "${CSX_REUSE_COMPLETED:?Run python -m scripts.audit_resume and export its audit path first}"
    prepare=reused
    bm25=reused
    reference=$(sbatch --parsable train-gpu.sbatch setting-a baselines-gpu encode-sample)
else
    prepare=$(sbatch --parsable train-cpu.sbatch setting-a prepare)
    bm25=$(sbatch --parsable --dependency="afterok:$prepare" train-cpu.sbatch setting-a baselines-cpu)
    reference=$(sbatch --parsable --dependency="afterok:$prepare" train-gpu.sbatch setting-a baselines-gpu encode-sample)
fi
fit=$(sbatch --parsable --dependency="afterok:$reference" train-cpu.sbatch setting-a fit-cpu)
encode=$(sbatch --parsable --dependency="afterok:$fit" train-gpu.sbatch setting-a fit-gpu encode-corpus)
dependencies="$encode"
[[ "$bm25" == reused ]] || dependencies="$bm25:$encode"
evaluate=$(sbatch --parsable --dependency="afterok:$dependencies" --export="ALL,SYNC_OUTPUT=${SYNC_OUTPUT:-0}" \
    train-cpu.sbatch setting-a join-encoded index evaluate analysis fit-predictor report)

printf 'prepare=%s bm25=%s reference=%s fit=%s encode=%s evaluate=%s\n' \
    "$prepare" "$bm25" "$reference" "$fit" "$encode" "$evaluate"
