#!/bin/bash
# from the repo root on a delta login node:
#   bash delta/submit.sh <account>        e.g. bxxx-delta-cpu (run `accounts` to see yours)
set -euo pipefail
ACCOUNT=${1:?usage: bash delta/submit.sh <account>}

for f in data/hcd_subset.h5ad data/hcd_subset_allgenes.h5ad data/collectri_human.csv; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
python delta/make_tasks.py > delta/tasks.tsv
N=$(grep -c . delta/tasks.tsv)

TRAIN=$(sbatch --parsable --account="$ACCOUNT" --array=0-$((N - 1)) delta/scgen_array.sbatch)
EVAL=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterok:"$TRAIN" delta/tf_eval.sbatch)
echo "scGen array: $TRAIN ($N tasks)"
echo "TF eval:     $EVAL (starts when all $N succeed)"
echo "check with:  squeue -u \$USER"
