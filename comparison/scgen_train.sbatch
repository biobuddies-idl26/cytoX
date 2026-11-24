#!/bin/bash
# Trains the 7 PBMC scGen models in parallel: one array task per held-out cell type.
# Submit from the repo's code/ folder with submit_scgen.sh (which also queues the reconstruction).
#
# Bridges-2 (PSC) settings below. For NCSA Delta instead, use:
#   #SBATCH --partition=cpu
#   #SBATCH --account=<your>-delta-cpu      (from the `accounts` command)
#   #SBATCH --mem=16g
#SBATCH --job-name=scgen-pbmc
#SBATCH --partition=RM-shared
#SBATCH --account=bio260074p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/train_%a_%x.log

CELL_TYPES=(NK CD4T CD14+Mono B CD8T FCGR3A+Mono Dendritic)
CT=${CELL_TYPES[$SLURM_ARRAY_TASK_ID]}

source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgen-repro
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Host: $(hostname) | cell type held out: $CT | start: $(date)"
python -u train_one.py "$CT"
echo "Finished $CT: $(date)"
