#!/bin/bash
# the TF-space steps (Model A scoring, TF prep, Model B, comparison) on one interactive
# node, for when the batch jobs sit in the queue. get a node first (1 hour max):
#   srun --account=bjhm-delta-cpu --partition=cpu-interactive --nodes=1 --tasks=1 \
#     --cpus-per-task=64 --mem=128g --time=01:00:00 --pty bash
# then from the repo root:  bash delta/run_now.sh
# safe to rerun: finished steps and Model B runs are skipped.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cytox
mkdir -p logs
NCPU=${SLURM_CPUS_PER_TASK:-8}
PAR=$((NCPU / 4))

for net in collectri shuffled; do
  [ -f "data/tf_scores_$net.h5ad" ] || OMP_NUM_THREADS=$NCPU python -u tf_scores.py --net "$net" || exit 1
done

[ -f delta/tasks_b.tsv ] || for net in collectri shuffled; do sed "s/^/$net\t/" delta/tasks.tsv; done > delta/tasks_b.tsv

# Model A in TF space, in the background while Model B trains
(
  [ -f data/scgen_runs/tf_results.csv ] || OMP_NUM_THREADS=8 python -u hcd_tf.py pred > logs/now_modelA_tf.log 2>&1
  [ -f data/scgen_runs/tf_results_shuffled.csv ] || OMP_NUM_THREADS=8 python -u hcd_tf.py pred --net shuffled > logs/now_modelA_tf_shuffled.log 2>&1
  echo "Model A TF scoring finished"
) &

run_b() {
  local out="data/model_b_runs/$2/task_$(printf %03d "$1")"
  [ -f "$out/results.csv" ] && return 0
  if OMP_NUM_THREADS=4 python -u model_b.py --cytokine "$3" --holdout "$4" \
      --scores "data/tf_scores_$2.h5ad" --outdir "$out" > "logs/now_model_b_$1.log" 2>&1; then
    echo "done $1 | $2 | $3 | $4"
  else
    echo "FAILED $1 | $2 | $3 | $4 (see logs/now_model_b_$1.log)"
  fi
}

# collectri runs come first in tasks_b.tsv, then shuffled
i=0
while IFS=$'\t' read -r net cyt ct; do
  while [ "$(jobs -rp | wc -l)" -gt "$PAR" ]; do wait -n; done
  run_b "$i" "$net" "$cyt" "$ct" &
  i=$((i + 1))
done < delta/tasks_b.tsv
wait

python - <<'PY'
import glob
import pandas as pd
files = sorted(glob.glob("data/scgen_runs/task_*/results.csv"))
pd.concat([pd.read_csv(f) for f in files]).to_csv("data/scgen_runs/results_all.csv", index=False)
PY
python -u compare_models.py | tee logs/now_compare.log
