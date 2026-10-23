#!/bin/bash
# Trains the four baseline models compared against scGen in Figure 2
# (PCA + vector arithmetic, vector arithmetic, style-transfer GAN, CVAE),
# all predicting CD4T responses. Runs alongside the scGen training array.
#
# For NCSA Delta: --partition=cpu, --account=<your>-delta-cpu, add --mem=16g
#SBATCH --job-name=scgen-baselines
#SBATCH --partition=RM-shared
#SBATCH --account=bio260074p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/baselines_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgen-repro
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
set -e

# Output folders the original scripts expect but never create
mkdir -p ../data/reconstructed/{PCAVecArithm,VecArithm,CGAN,CVAE} \
         ../models/STGAN ../models/CVAE/pbmc/all/models \
         "../results/Figures/Supplemental Figure 4"

echo "== PCA vector arithmetic: $(date)";  python -u vec_arith_pca.py
echo "== Vector arithmetic: $(date)";      python -u vec_arith.py
echo "== Style-transfer GAN: $(date)";     python -u st_gan.py train
echo "== CVAE: $(date)";                   python -u train_cvae.py

# Two scripts save under different names than Fig2.ipynb reads; copy them to the expected paths
cp ../data/reconstructed/CGAN/cgan_cd4t.h5ad ../data/reconstructed/CGAN/CGAN_CD4T.h5ad
cp ../data/reconstructed/CVAE_CD4T.h5ad      ../data/reconstructed/CVAE/CVAE_CD4T.h5ad

echo "== Baselines done: $(date)"
ls -lh ../data/reconstructed/{PCAVecArithm,VecArithm,CGAN,CVAE}/
