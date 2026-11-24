"""Compare the original scGen and shared scGen; require identical naive scores."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path, help="comparison root, or shared output directory with --historical")
p.add_argument("--historical", action="store_true", help="compare with bundled original results instead of rerunning")
p.add_argument("--original-run", default="scgen_runs_L2_H800_Z100", help="run name in the historical results CSV")
args = p.parse_args()
keys = ["cytokine", "holdout", "model"]
metrics = ["r_all", "mse", "r_top100"]
if args.historical:
    b = pd.read_csv(args.root / "results.csv")
    history = Path(__file__).resolve().parent / "analysis/results_L2_vs_L4_H800_Z100.csv"
    a = pd.read_csv(history)
    a = a[a.run == args.original_run].copy()
    a = a.merge(b[["cytokine", "holdout"]].drop_duplicates(), on=["cytokine", "holdout"], how="inner")
else:
    a = pd.read_csv(args.root / "original/results.csv")
    b = pd.read_csv(args.root / "shared/results.csv")

assert not a.duplicated(keys).any() and not b.duplicated(keys).any(), "Duplicate results"
c = a[keys + metrics].merge(b[keys + metrics], on=keys, suffixes=("_original", "_shared"), how="outer", validate="one_to_one", indicator=True)
assert (c["_merge"] == "both").all(), "Mismatched experiments"
c = c.drop(columns="_merge")
assert set(c.model) == {"scGen", "scGen_vs_real_ctrl", "naive"}, "Unexpected methods"
for m in metrics:
    c[m + "_shared_minus_original"] = c[m + "_shared"] - c[m + "_original"]
n = c[c.model == "naive"]
assert len(n) == len(c) // 3 and len(n) > 0, "Missing naive scores"
match = np.ones(len(n), dtype=bool)
for m in metrics:
    match &= np.isclose(n[m + "_original"], n[m + "_shared"], atol=1e-7, rtol=0, equal_nan=False)
if not match.all():
    print(n.loc[~match].to_string(index=False))
    raise SystemExit("Naive metrics differ; verify the exact original H5AD (including cells and genes). No matched comparison written.")
c.to_csv(args.root / "comparison.csv", index=False)
print("Naive baseline matches in both runs at the original CSV precision.")
print(c.to_string(index=False))
print("\nMean metrics by model:")
print(c.groupby("model")[[m+s for m in metrics for s in ("_original", "_shared")]].mean().to_string())
