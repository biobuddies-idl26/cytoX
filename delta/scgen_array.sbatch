#!/bin/bash
# one scGen run per line of delta/tasks.tsv.
# architecture via --export, e.g.: sbatch --export=ALL,NL=4 ... delta/scgen_array.sbatch
#SBATCH --job-name=cytox-scgen
#SBATCH --account=bjhm-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=03:00:00
#SBATCH --output=logs/scgen_%A_%a.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -euo pipefail

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" delta/tasks.tsv)
CYT=$(echo "$LINE" | cut -f1)
CT=$(echo "$LINE" | cut -f2)

# architecture (defaults = original scGen paper: 2 layers x 800, 100-d latent)
NL=${NL:-2}; NH=${NH:-800}; NZ=${NZ:-100}
echo "task $SLURM_ARRAY_TASK_ID | $CYT | held out: $CT | L=$NL H=$NH Z=$NZ | $(hostname) | $(date)"

# W&B: compute nodes have internet, so runs log live
export WANDB_DIR=/projects/bjhm/$USER/wandb
mkdir -p "$WANDB_DIR"

# own folder per architecture and per task, so runs never overwrite each other
python -u scgen_baseline.py --cytokine "$CYT" --holdout "$CT" \
  --n-layers "$NL" --n-hidden "$NH" --n-latent "$NZ" \
  --wandb-project cytoX --wandb-group "L${NL}_H${NH}_Z${NZ}" \
  --outdir "data/scgen_runs_L${NL}_H${NH}_Z${NZ}/task_$(printf %03d "$SLURM_ARRAY_TASK_ID")"
echo "done $(date)"
