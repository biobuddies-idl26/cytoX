#!/bin/bash
# Builds data/reconstructed/scGen/pbmc.h5ad from all 7 trained models (input for Fig2.ipynb).
# Queued automatically by submit_scgen.sh to run only if every training task succeeds.
#
# For NCSA Delta: --partition=cpu, --account=<your>-delta-cpu, add --mem=16g
#SBATCH --job-name=scgen-recon
#SBATCH --partition=RM-shared
#SBATCH --account=bio260074p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/reconstruct_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgen-repro

python -u -c "from train_scGen import reconstruct_whole_data; reconstruct_whole_data('pbmc')"
ls -lh ../data/reconstructed/scGen/pbmc.h5ad
