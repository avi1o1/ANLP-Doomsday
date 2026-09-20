#!/bin/bash
set -euo pipefail

# Prepare once, overlap the independent CPU BM25S and GPU reference work, then
# join both branches before evaluation. --parsable returns only each Slurm job ID.
prepare=$(sbatch --parsable train-cpu.sbatch setting-a prepare)
bm25=$(sbatch --parsable --dependency="afterok:$prepare" train-cpu.sbatch setting-a baselines-cpu)
reference=$(sbatch --parsable --dependency="afterok:$prepare" train-gpu.sbatch setting-a baselines-gpu encode-sample)
fit=$(sbatch --parsable --dependency="afterok:$reference" train-cpu.sbatch setting-a fit-cpu)
encode=$(sbatch --parsable --dependency="afterok:$fit" train-gpu.sbatch setting-a fit-gpu encode-corpus)
evaluate=$(sbatch --parsable --dependency="afterok:$bm25:$encode" --export=ALL,SYNC_OUTPUT=1 \
    train-cpu.sbatch setting-a index evaluate analysis fit-predictor report)

printf 'prepare=%s bm25=%s reference=%s fit=%s encode=%s evaluate=%s\n' \
    "$prepare" "$bm25" "$reference" "$fit" "$encode" "$evaluate"
