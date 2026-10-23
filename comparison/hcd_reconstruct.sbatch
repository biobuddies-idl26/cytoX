#!/bin/bash
# Runs after all hcd-train tasks succeed: builds predictions for every cytokine, then scores them.
#SBATCH --job-name=hcd-recon
#SBATCH --partition=RM-shared
#SBATCH --account=bio260074p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/hcd_reconstruct_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgen-repro
set -e

for CYT in $(cut -f1 ../data/hcd/tasks.tsv | sort -u); do
  python -u hcd_scgen.py reconstruct "$CYT"
done

for CYT in $(cut -f1 ../data/hcd/tasks.tsv | sort -u); do
  python -u evaluate_r2.py ../data/reconstructed/scGen/hcd_${CYT}.h5ad
done
