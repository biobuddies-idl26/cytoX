#!/usr/bin/env bash
# Train original and shared scGen on exactly the same H5AD and task list.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${1:-data/hcd_subset.h5ad}
HOLDOUT=${2:-CD14 Mono}
ROOT=${3:-data/shared_comparison}
EPOCHS=${EPOCHS:-100}
NL=${NL:-2}; NH=${NH:-800}; NZ=${NZ:-100}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mapfile -t CYTS < <(awk -F '\t' -v ct="$HOLDOUT" '$2==ct {print $1}' delta/tasks.tsv)
if [[ ${#CYTS[@]} -eq 0 ]]; then echo "Unknown holdout: $HOLDOUT" >&2; exit 1; fi
TAG=${HOLDOUT// /_}; TAG=${TAG//\//-}
OUT="$ROOT/L${NL}_H${NH}_Z${NZ}/$TAG"
mkdir -p "$OUT/logs"
if [[ -e "$OUT/shared/results.csv" || -e "$OUT/original/results.csv" ]]; then
  echo "Existing results in $OUT; use a new output root." >&2; exit 1
fi
for CYT in "${CYTS[@]}"; do
  python -u scgen_baseline.py --data "$DATA" --holdout "$HOLDOUT" --cytokine "$CYT" \
    --epochs "$EPOCHS" --n-layers "$NL" --n-hidden "$NH" --n-latent "$NZ" \
    --outdir "$OUT/original" 2>&1 | tee "$OUT/logs/original_${CYT}.log"
done
python -u scgen_shared.py --data "$DATA" --holdout "$HOLDOUT" --cytokines "${CYTS[@]}" \
  --epochs "$EPOCHS" --n-layers "$NL" --n-hidden "$NH" --n-latent "$NZ" \
  --outdir "$OUT/shared" 2>&1 | tee "$OUT/logs/shared.log"
python compare_shared.py "$OUT"
