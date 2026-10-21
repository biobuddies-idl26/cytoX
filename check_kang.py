# sanity check for collecTRI.py on Kang et al. 2018 (PBMCs +/- IFN-beta).
# IFN-beta signals through JAK-STAT -> ISGF3 (STAT1/STAT2/IRF9), so those TFs
# should come out strongly positive in stim vs ctrl for basically every cell type.

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import scanpy as sc

from collecTRI import load_collectri, filter_net, tf_scores, delta_tf, compare

KANG_URL = "https://exampledata.scverse.org/pertpy/kang_2018.h5ad"
KANG_PATH = "data/kang_2018.h5ad"
N_HVG = 7000
EXPECTED_UP = ["STAT1", "STAT2", "IRF9", "IRF1"]

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
if not os.path.exists(KANG_PATH):
    # requests uses certifi's certs, so this works on python.org mac installs too
    r = requests.get(KANG_URL, timeout=120)
    r.raise_for_status()
    with open(KANG_PATH, "wb") as f:
        f.write(r.content)

adata = sc.read_h5ad(KANG_PATH)
adata = adata[adata.obs["cell_type"] != "Megakaryocytes"].copy()  # too few cells per donor
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.shape, adata.obs["label"].value_counts().to_dict())

net = load_collectri()
print("collectri:", net["source"].nunique(), "TFs,", len(net), "edges")

# all genes vs HVGs (scGen-style input). same net restriction is applied to
# anything we compare, so TF sets line up.
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
hvg = adata.var_names[adata.var["highly_variable"]]

results = {}
for name, genes in [("all", adata.var_names), ("hvg", hvg)]:
    sub = adata[:, genes]
    n = filter_net(net, genes)
    print(f"[{name}] {len(genes)} genes -> {n['source'].nunique()} TFs with >=5 targets")
    s = tf_scores(sub, n)
    d = delta_tf(s, sub.obs, "label", "ctrl", "stim", "cell_type", donor_key="replicate")
    d.to_csv(f"data/kang_delta_tf_{name}.csv")
    results[name] = d

d = results["all"]
print("\nmean delta TF across cell types, top 10:")
print(d.mean().sort_values(ascending=False).head(10).round(2))
print("\nexpected-up TFs per cell type:")
print(d[[t for t in EXPECTED_UP if t in d.columns]].round(2))

ok = (d[[t for t in EXPECTED_UP if t in d.columns]] > 0).all().all()
print("\nPASS" if ok else "\nFAIL", "- all expected TFs up in every cell type")

print("\nhow much do we lose by scoring on HVGs only? (hvg vs all-genes delta)")
print(compare(results["hvg"], results["all"]).round(3))

# heatmap of top TFs
top = d.abs().mean().sort_values(ascending=False).head(20).index
v = np.abs(d[top].values).max()
fig, ax = plt.subplots(figsize=(9, 3.6))
im = ax.imshow(d[top].values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(len(top)), top, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(d)), d.index, fontsize=8)
ax.set_title("Kang 2018: TF activity change, IFN-$\\beta$ stim - ctrl (ULM, CollecTRI)", fontsize=9)
fig.colorbar(im, ax=ax, label="$\\Delta$ ULM score")
fig.tight_layout()
fig.savefig("figures/kang_delta_tf_heatmap.png", dpi=200)
print("saved figures/kang_delta_tf_heatmap.png")
