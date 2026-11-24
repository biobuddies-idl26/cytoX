"""Train one scGen PBMC model, holding out the stimulated cells of one cell type.

Usage (from code/):  python -u train_one.py CD4T
Hyperparameters match the original train_scGen.py exactly.
"""
import os
import sys

from train_scGen import test_train_whole_data_one_celltype_out

cell_type = sys.argv[1]

# The original code saves on early stopping before creating its folder,
# so make sure the folders exist first.
os.makedirs(f"../models/scGen/pbmc/{cell_type}", exist_ok=True)
os.makedirs("../data/reconstructed/scGen", exist_ok=True)

test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                       dropout_rate=0.2, learning_rate=0.001,
                                       cell_type_to_train=cell_type)
