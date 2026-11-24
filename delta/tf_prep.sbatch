#!/bin/bash
# TF activity for every cell, real and shuffled network. Model B trains on these.
#SBATCH --job-name=cytox-tfprep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=logs/tf_prep_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -euo pipefail

python -u tf_scores.py --net collectri
python -u tf_scores.py --net shuffled
