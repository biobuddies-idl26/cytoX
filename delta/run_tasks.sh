#!/bin/bash
# usage: bash delta/run_tasks.sh <n_layers>   (skips tasks that already have results)
NL=${1:-2}
ROOT=data/scgen_runs_L${NL}_H800_Z100
i=0
while IFS=$'\t' read -r CYT CT; do
  OUT="$ROOT/task_$(printf %03d $i)"
  if [ ! -f "$OUT/results.csv" ]; then
    echo "=== L=$NL task $i | $CYT | $CT | $(date +%T)"
    python -u scgen_baseline.py --cytokine "$CYT" --holdout "$CT" --n-layers "$NL" \
      --wandb-project cytoX --wandb-group "L${NL}_H800_Z100" --outdir "$OUT" \
      || echo "FAILED task $i"
  fi
  i=$((i+1))
done < delta/tasks.tsv
echo "L=$NL: $(ls $ROOT/*/results.csv 2>/dev/null | wc -l) / $(grep -c . delta/tasks.tsv) done"
