#!/bin/bash
# runs after every scGen task finishes: merges the gene-space scores, then collecTRI truth + pred.
#SBATCH --job-name=cytox-tf
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=logs/tf_eval_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -euo pipefail

python - <<'PY'
import glob
import pandas as pd
files = sorted(glob.glob("data/scgen_runs/task_*/results.csv"))
df = pd.concat([pd.read_csv(f) for f in files])
df.to_csv("data/scgen_runs/results_all.csv", index=False)
print(f"{len(files)} runs merged -> data/scgen_runs/results_all.csv")
PY

python -u hcd_tf.py truth
python -u hcd_tf.py pred
