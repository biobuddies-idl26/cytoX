#!/bin/bash
# One task per line of ../data/hcd/tasks.tsv (cytokine x held-out cell type).
# The array size is set by submit_hcd.sh, so don't add --array here.
#SBATCH --job-name=hcd-train
#SBATCH --partition=RM-shared
#SBATCH --account=bio260074p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/hcd_train_%a.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgen-repro
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ../data/hcd/tasks.tsv)
CYT=$(echo "$LINE" | cut -f1)
CT=$(echo "$LINE" | cut -f2)

echo "Task $SLURM_ARRAY_TASK_ID | cytokine: $CYT | held-out cell type: $CT | $(hostname) | $(date)"
python -u hcd_scgen.py train "$CYT" "$CT"
echo "Finished: $(date)"
