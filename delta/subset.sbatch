#!/bin/bash
#SBATCH --job-name=hcd-subset-n5000
#SBATCH --account=bjhm-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=06:00:00
#SBATCH --output=logs/subset_n5000_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
set -euo pipefail

RAW=/projects/bjhm/data/hcd/raw/20260203_Parse_10M_PBMC_cytokines.h5ad
OUT=/projects/bjhm/data/hcd/processed/n5000
N=5000
CYTS="PBS IFN-beta IFN-gamma IL-4 IL-15 IL-1-beta IL-32-beta"   # confirm with Neha

mkdir -p "$OUT"
echo "start $(date) | n=$N | cytokines: $CYTS"

# 7,000 HVGs: what scGen trains on
python -u subset_hcd.py --raw "$RAW" --cytokines $CYTS --n $N \
  --out "$OUT/hcd_subset.h5ad"

# all genes: needed later for CollecTRI / TF activity (Model B)
python -u subset_hcd.py --raw "$RAW" --cytokines $CYTS --n $N --all-genes \
  --out "$OUT/hcd_subset_allgenes.h5ad"

chmod -R g+rwX "$OUT"
echo "done $(date)"
ls -lh "$OUT"
