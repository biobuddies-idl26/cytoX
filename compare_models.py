# one table with every model scored on delta TF, for the real and shuffled network.
#   Model A  = scGen on genes -> TF inferred from predictions   (hcd_tf.py pred)
#   Model B  = scGen on TF activity                               (model_b.py)
#   naive_tf = mean delta TF of the training cell types           (model_b.py)
# python compare_models.py

import glob
import os

import pandas as pd

rows = []
for net in ["collectri", "shuffled"]:
    f = "data/scgen_runs/tf_results.csv" if net == "collectri" else f"data/scgen_runs/tf_results_{net}.csv"
    if os.path.exists(f):
        a = pd.read_csv(f).assign(net=net, model="modelA")
        rows.append(a)
for f in glob.glob("data/model_b_runs/**/results.csv", recursive=True):
    rows.append(pd.read_csv(f))
if not rows:
    raise SystemExit("no results yet")

df = pd.concat(rows, ignore_index=True)
df = df[df["model"] != "modelB_vs_real_ctrl"]
df.to_csv("data/all_tf_results.csv", index=False)

metrics = ["pearson", "mse", "sign_agree"]
overall = df.groupby(["net", "model"])[metrics].agg(["mean", "std"]).round(3)
overall[("n", "")] = df.groupby(["net", "model"]).size()
print("mean over all cytokine x held-out cell type combinations:\n")
print(overall.to_string())

per_cyto = df.pivot_table(index=["net", "cytokine"], columns="model", values="pearson").round(3)
print("\npearson r of predicted vs real delta TF, by cytokine:\n")
print(per_cyto.to_string())

# head to head on the same combinations
w = df.pivot_table(index=["net", "cytokine", "holdout"], columns="model", values="pearson")
print("\nfraction of combinations where the first model has higher r:")
for net in w.index.get_level_values("net").unique():
    x = w.loc[net]
    for a, b in [("modelB", "modelA"), ("modelA", "naive_tf"), ("modelB", "naive_tf")]:
        if a in x and b in x:
            both = x[[a, b]].dropna()
            print(f"  {net:<10} {a} > {b}: {(both[a] > both[b]).mean():.2f}  (n={len(both)})")

overall.to_csv("data/summary_overall.csv")
per_cyto.to_csv("data/summary_by_cytokine.csv")
print("\nsaved data/all_tf_results.csv, data/summary_overall.csv, data/summary_by_cytokine.csv")
