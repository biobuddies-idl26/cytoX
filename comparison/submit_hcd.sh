#!/bin/bash
# Run from code/ after prepare_hcd.py:  bash submit_hcd.sh
set -euo pipefail
mkdir -p logs

N=$(grep -c . ../data/hcd/tasks.tsv)
TRAIN_ID=$(sbatch --parsable --array=0-$((N - 1)) hcd_train.sbatch)
echo "HCD training array submitted: $TRAIN_ID ($N tasks)"

RECON_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID hcd_reconstruct.sbatch)
echo "Reconstruct + score queued:   $RECON_ID (starts after all $N succeed)"
echo "Check status with:            squeue -u \$USER"
