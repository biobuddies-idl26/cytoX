"""Utilities for constructing HCD pseudobulk AnnData objects.

The primary grouping is donor x cell_type x cytokine.  Two expression
representations are supported:

* ``sum_counts``: sum raw UMI counts within each group, then library-size
  normalize to ``target_sum`` and apply log1p.  This is the conventional
  count-based pseudobulk representation.
* ``mean_log``: average the existing log-normalized cell-level expression
  values within each group.  This is an ML-oriented comparison, not a bulk
  count matrix.

A third ``pooled_donor`` grouping collapses donors and is included as an
exploratory comparison only.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


def _as_csr(matrix):
    """Return a CSR matrix without unnecessarily densifying sparse inputs."""
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(np.asarray(matrix))


def _row_sum(matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1)).ravel()


def _normalise_sum_counts(count_matrix, target_sum: float = 1e4):
    """Library-size normalize summed counts and apply log1p."""
    counts = _as_csr(count_matrix).astype(np.float32)
    totals = _row_sum(counts).astype(np.float64)
    if np.any(totals <= 0):
        bad = np.flatnonzero(totals <= 0).tolist()
        raise ValueError(f"Pseudobulk groups have zero total counts: rows {bad}")
    scale = (float(target_sum) / totals).astype(np.float32)
    normalized = sp.diags(scale).dot(counts).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized.astype(np.float32), counts


def _mean_rows(matrix, indices: np.ndarray):
    """Mean expression across selected rows, preserving sparse storage."""
    selected = _as_csr(matrix)[indices]
    return selected.mean(axis=0)


def _group_cell_counts(obs: pd.DataFrame, keys: list[str]) -> pd.Series:
    return obs.groupby(keys, observed=True, sort=True).size()


def _eligible_donor_groups(
    obs: pd.DataFrame,
    cytokine: str,
    min_cells: int,
    control_label: str = "PBS",
) -> set[tuple[str, str]]:
    """Return donor/cell-type pairs with enough cells in BOTH conditions."""
    counts = _group_cell_counts(obs, ["donor", "cell_type", "cytokine"])
    eligible: set[tuple[str, str]] = set()
    for (donor, cell_type), _ in obs.groupby(["donor", "cell_type"], observed=True):
        n_control = int(counts.get((donor, cell_type, control_label), 0))
        n_stim = int(counts.get((donor, cell_type, cytokine), 0))
        if n_control >= min_cells and n_stim >= min_cells:
            eligible.add((str(donor), str(cell_type)))
    return eligible


def _eligible_pooled_cell_types(
    obs: pd.DataFrame,
    cytokine: str,
    min_cells: int,
    control_label: str = "PBS",
) -> set[str]:
    """Return cell types with enough pooled cells in both conditions."""
    counts = _group_cell_counts(obs, ["cell_type", "cytokine"])
    eligible = set()
    for cell_type in obs["cell_type"].astype(str).unique():
        if (int(counts.get((cell_type, control_label), 0)) >= min_cells and
                int(counts.get((cell_type, cytokine), 0)) >= min_cells):
            eligible.add(str(cell_type))
    return eligible


def build_pseudobulk(
    adata: ad.AnnData,
    cytokine: str,
    strategy: str = "sum_counts",
    min_cells: int = 10,
    control_label: str = "PBS",
    target_sum: float = 1e4,
    donor_key: str = "donor",
    cell_type_key: str = "cell_type",
    cytokine_key: str = "cytokine",
    counts_layer: str = "counts",
) -> ad.AnnData:
    """Build pseudobulk profiles for one cytokine and its PBS controls.

    For donor-specific strategies, a donor x cell-type pair is retained only
    when BOTH PBS and the requested cytokine have at least ``min_cells``.
    This paired eligibility rule ensures the same donor population contributes
    to the treatment and control profiles for a given cell type.

    Parameters
    ----------
    adata:
        Cell-level AnnData. ``adata.X`` must contain log1p-normalized expression
        for ``mean_log``. ``adata.layers[counts_layer]`` must contain raw counts
        for ``sum_counts`` and ``pooled_donor``.
    cytokine:
        Cytokine condition to compare against PBS.
    strategy:
        ``sum_counts``, ``mean_log``, or ``pooled_donor``.
    min_cells:
        Minimum cells required in each condition. For donor-specific strategies
        this is applied to each donor x cell type x condition group. For pooled
        donor it is applied to each cell type x condition group.
    """
    if strategy not in {"sum_counts", "mean_log", "pooled_donor"}:
        raise ValueError(f"Unknown pseudobulk strategy: {strategy}")
    if min_cells < 1:
        raise ValueError("min_cells must be >= 1")
    required = {donor_key, cell_type_key, cytokine_key}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {sorted(missing)}")
    if strategy in {"sum_counts", "pooled_donor"} and counts_layer not in adata.layers:
        raise KeyError(
            f"Raw-count layer '{counts_layer}' not found. Available layers: "
            f"{list(adata.layers.keys())}"
        )

    LOGGER.info("Building pseudobulk: strategy=%s, cytokine=%s, min_cells=%d",
                strategy, cytokine, min_cells)

    # Restrict to the requested treatment and PBS before grouping.
    cytokine_values = adata.obs[cytokine_key].astype(str)
    keep = cytokine_values.isin([control_label, cytokine]).to_numpy()
    cell = adata[keep].copy()
    obs = cell.obs.copy()
    for key in (donor_key, cell_type_key, cytokine_key):
        obs[key] = obs[key].astype(str)

    present = set(obs[cytokine_key].unique())
    if control_label not in present or cytokine not in present:
        raise ValueError(
            f"Need both '{control_label}' and '{cytokine}' in the selected data; "
            f"found {sorted(present)}"
        )

    if strategy == "pooled_donor":
        eligible_cell_types = _eligible_pooled_cell_types(
            obs.rename(columns={cell_type_key: "cell_type", cytokine_key: "cytokine"}),
            cytokine, min_cells, control_label
        )
        eligible_mask = obs[cell_type_key].isin(eligible_cell_types).to_numpy()
        grouping_keys = [cell_type_key, cytokine_key]
        LOGGER.info("Pooled-donor grouping: %d eligible cell types", len(eligible_cell_types))
    else:
        eligible_pairs = _eligible_donor_groups(
            obs.rename(columns={donor_key: "donor", cell_type_key: "cell_type",
                                cytokine_key: "cytokine"}),
            cytokine, min_cells, control_label
        )
        pair_values = list(zip(obs[donor_key].astype(str), obs[cell_type_key].astype(str)))
        eligible_mask = np.fromiter((pair in eligible_pairs for pair in pair_values),
                                    dtype=bool, count=len(pair_values))
        grouping_keys = [donor_key, cell_type_key, cytokine_key]
        LOGGER.info("Paired donor/cell-type groups retained: %d", len(eligible_pairs))

    cell = cell[eligible_mask].copy()
    obs = cell.obs.copy()
    for key in (donor_key, cell_type_key, cytokine_key):
        obs[key] = obs[key].astype(str)

    if cell.n_obs == 0:
        raise ValueError(
            f"No cells remain after paired filtering (strategy={strategy}, "
            f"min_cells={min_cells}, cytokine={cytokine})"
        )

    if strategy == "pooled_donor":
        matrix_keys = [cell_type_key, cytokine_key]
    else:
        matrix_keys = [donor_key, cell_type_key, cytokine_key]

    grouped = obs.groupby(matrix_keys, observed=True, sort=True).indices
    counts_matrix = _as_csr(cell.layers[counts_layer]) if strategy != "mean_log" else None
    log_matrix = _as_csr(cell.X) if strategy == "mean_log" else None

    output_rows = []
    count_rows = []
    metadata = []
    for group_key, row_indices in grouped.items():
        row_indices = np.asarray(row_indices, dtype=int)
        group_obs = obs.iloc[row_indices]
        n_cells = len(row_indices)

        if strategy in {"sum_counts", "pooled_donor"}:
            summed = counts_matrix[row_indices].sum(axis=0)
            summed = sp.csr_matrix(summed, dtype=np.float32)
            count_rows.append(summed)
            output_rows.append(summed)
        else:
            # Average the already log-normalized cell-level expression.
            mean_row = log_matrix[row_indices].mean(axis=0)
            output_rows.append(sp.csr_matrix(mean_row, dtype=np.float32))
            # Keep summed counts as a useful audit trail, although X is mean-log.
            summed = _as_csr(cell.layers[counts_layer][row_indices]).sum(axis=0)
            count_rows.append(sp.csr_matrix(summed, dtype=np.float32))

        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        if strategy == "pooled_donor":
            cell_type, condition_cytokine = group_key
            donor_value = "POOLED"
            n_donors = group_obs[donor_key].nunique()
        else:
            donor_value, cell_type, condition_cytokine = group_key
            n_donors = 1

        metadata.append({
            "donor": str(donor_value),
            "cell_type": str(cell_type),
            "cytokine": str(condition_cytokine),
            "condition": "control" if str(condition_cytokine) == control_label else "stimulated",
            "n_cells": int(n_cells),
            "n_donors": int(n_donors),
            "strategy": strategy,
            "min_cells": int(min_cells),
        })

    # Sum-count strategies are normalized AFTER aggregation. Mean-log profiles
    # already use the subset_hcd.py normalized/log1p X matrix.
    summed_counts = sp.vstack(count_rows, format="csr", dtype=np.float32)
    if strategy in {"sum_counts", "pooled_donor"}:
        X, summed_counts = _normalise_sum_counts(summed_counts, target_sum)
    else:
        X = sp.vstack(output_rows, format="csr", dtype=np.float32)

    pb_obs = pd.DataFrame(metadata)
    pb_obs.index = [f"pb_{i:06d}" for i in range(len(pb_obs))]
    pb_var = cell.var.copy()
    result = ad.AnnData(X=X, obs=pb_obs, var=pb_var)
    result.layers["counts"] = summed_counts
    result.uns["pseudobulk"] = {
        "strategy": strategy,
        "min_cells": int(min_cells),
        "target_sum": float(target_sum),
        "control_label": control_label,
        "cytokine": cytokine,
        "grouping": "cell_type x cytokine" if strategy == "pooled_donor"
                    else "donor x cell_type x cytokine",
        "expression_semantics": "mean_log_expression" if strategy == "mean_log"
                                else "sum_counts_then_normalize_log1p",
    }

    LOGGER.info("Created %d pseudobulk profiles from %d cells (%d genes)",
                result.n_obs, cell.n_obs, result.n_vars)
    LOGGER.info("Pseudobulk cell-count summary: min=%d median=%.1f max=%d",
                int(result.obs["n_cells"].min()),
                float(result.obs["n_cells"].median()),
                int(result.obs["n_cells"].max()))
    LOGGER.info("Profiles by condition: %s",
                result.obs["condition"].value_counts().to_dict())
    return result


def paired_donor_summary(
    pseudobulk: ad.AnnData,
    cytokine: str,
    holdout: Optional[str] = None,
) -> pd.DataFrame:
    """Return per-cell-type counts of donors with paired PBS/cytokine profiles."""
    obs = pseudobulk.obs
    if "donor" not in obs or (obs["donor"].astype(str) == "POOLED").all():
        return pd.DataFrame()
    subset = obs if holdout is None else obs[obs["cell_type"].astype(str) == holdout]
    rows = []
    for cell_type, group in subset.groupby("cell_type", observed=True):
        donors_control = set(group.loc[group["cytokine"] == "PBS", "donor"].astype(str))
        donors_stim = set(group.loc[group["cytokine"] == cytokine, "donor"].astype(str))
        paired = donors_control & donors_stim
        rows.append({
            "cell_type": str(cell_type),
            "n_control_profiles": int((group["cytokine"] == "PBS").sum()),
            "n_stim_profiles": int((group["cytokine"] == cytokine).sum()),
            "n_paired_donors": len(paired),
        })
    return pd.DataFrame(rows)
