#!/bin/bash
# one Model B run per line of delta/tasks_b.tsv (net, cytokine, held-out cell type).
# submit with delta/submit_b.sh, not directly.
#SBATCH --job-name=cytox-modelb
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=logs/model_b_%a.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -euo pipefail

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" delta/tasks_b.tsv)
NET=$(echo "$LINE" | cut -f1)
CYT=$(echo "$LINE" | cut -f2)
CT=$(echo "$LINE" | cut -f3)
echo "task $SLURM_ARRAY_TASK_ID | $NET | $CYT | held out: $CT | $(hostname) | $(date)"

python -u model_b.py --cytokine "$CYT" --holdout "$CT" --scores "data/tf_scores_${NET}.h5ad" \
  --outdir "data/model_b_runs/${NET}/task_$(printf %03d "$SLURM_ARRAY_TASK_ID")"
echo "done $(date)"
