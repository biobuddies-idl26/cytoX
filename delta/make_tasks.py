# one line per (cytokine, held-out cell type) for the delta array job.
# python delta/make_tasks.py > delta/tasks.tsv

import sys

import anndata as ad

RARE = ["pDC", "HSPC", "ILC", "Plasmablast"]  # same exclusion as scgen_baseline.py

data = sys.argv[1] if len(sys.argv) > 1 else "data/hcd_subset.h5ad"
obs = ad.read_h5ad(data, backed="r").obs
cytokines = sorted(set(obs["cytokine"].astype(str)) - {"PBS"})
cell_types = sorted(set(obs["cell_type"].astype(str)) - set(RARE))
for c in cytokines:
    for ct in cell_types:
        print(f"{c}\t{ct}")
