#!/bin/bash
# one scGen run per line of delta/tasks.tsv. submit with delta/submit.sh, not directly.
#SBATCH --job-name=cytox-scgen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=logs/scgen_%a.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -euo pipefail

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" delta/tasks.tsv)
CYT=$(echo "$LINE" | cut -f1)
CT=$(echo "$LINE" | cut -f2)
echo "task $SLURM_ARRAY_TASK_ID | $CYT | held out: $CT | $(hostname) | $(date)"

# own folder per task so parallel runs don't write to the same results.csv
python -u scgen_baseline.py --cytokine "$CYT" --holdout "$CT" \
  --outdir "data/scgen_runs/task_$(printf %03d "$SLURM_ARRAY_TASK_ID")"
echo "done $(date)"
