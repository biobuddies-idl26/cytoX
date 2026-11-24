#!/bin/bash
# Model B + ablations, from the repo root on a delta login node:
#   bash delta/submit_b.sh <account> [model A TF job id]
# with the job id, the final comparison waits for Model A's tf_eval job too.
set -euo pipefail
ACCOUNT=${1:?usage: bash delta/submit_b.sh <account> [model A TF job id]}
AFTER_A=${2:-}

[ -f delta/tasks.tsv ] || { echo "missing delta/tasks.tsv, run delta/submit.sh first"; exit 1; }
mkdir -p logs
for net in collectri shuffled; do
  sed "s/^/$net\t/" delta/tasks.tsv
done > delta/tasks_b.tsv
N=$(grep -c . delta/tasks_b.tsv)

PREP=$(sbatch --parsable --account="$ACCOUNT" delta/tf_prep.sbatch)
B=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterok:"$PREP" --array=0-$((N - 1)) delta/model_b_array.sbatch)
DEP="afterok:$B"
[ -n "$AFTER_A" ] && DEP="$DEP:$AFTER_A"
CMP=$(sbatch --parsable --account="$ACCOUNT" --dependency="$DEP" delta/compare.sbatch)
echo "TF scores:   $PREP"
echo "Model B:     $B ($N tasks, starts after $PREP)"
echo "comparison:  $CMP (starts after Model B${AFTER_A:+ and $AFTER_A})"
