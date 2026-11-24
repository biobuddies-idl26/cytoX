#!/bin/bash
# Run from the repo's code/ folder:  bash submit_scgen.sh
set -euo pipefail
mkdir -p logs

TRAIN_ID=$(sbatch --parsable scgen_train.sbatch)
echo "Training array submitted: $TRAIN_ID (7 tasks)"

RECON_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID scgen_reconstruct.sbatch)
echo "Reconstruction queued:    $RECON_ID (starts after all 7 succeed)"
echo "Check status with:        squeue -u \$USER"
