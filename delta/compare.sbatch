#!/bin/bash
# after Model A and Model B finish: rescore Model A with the shuffled network, then one summary table.
#SBATCH --job-name=cytox-compare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=03:00:00
#SBATCH --output=logs/compare_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -euo pipefail

python -u hcd_tf.py pred --net shuffled
python -u compare_models.py
