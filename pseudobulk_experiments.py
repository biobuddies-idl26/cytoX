"""Run scGen-style held-out-cell-type prediction on HCD pseudobulk profiles.

Example smoke test:
    python pseudobulk_experiments.py --data data/hcd_subset.h5ad \
        --cytokines IFN-beta --holdouts "CD14 Mono" \
        --strategies sum_counts --thresholds 10 --epochs 5 --no-wandb

Full comparison (can launch many runs):
    python pseudobulk_experiments.py --data data/hcd_subset.h5ad \
        --cytokines IFN-beta IFN-gamma IL-4 IL-15 IL-1-beta IL-32-beta \
        --strategies sum_counts mean_log pooled_donor \
        --thresholds 5 10 20 --pooled-min-cells 10

The existing scgen_baseline.py remains the cell-level baseline and is not
modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pseudobulk import build_pseudobulk, paired_donor_summary

# ---------------------------------------------------------------------------
# EDIT THESE THREE VALUES FOR YOUR WEIGHTS & BIASES WORKSPACE.
# You can also override project/entity from the command line.
# ---------------------------------------------------------------------------
WANDB_PROJECT = "cytoX"
WANDB_ENTITY = "scgen"          # e.g. "your-lab"; None uses your default W&B entity
WANDB_MODE = "online"       # set to "offline" for local W&B buffering

RARE_CELL_TYPES = ["pDC", "HSPC", "ILC", "Plasmablast"]
DEFAULT_STRATEGIES = ["sum_counts", "mean_log", "pooled_donor"]
DEFAULT_THRESHOLDS = [5, 10, 20]


def configure_logging(outdir: Path, log_level: str = "INFO") -> logging.Logger:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pseudobulk_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logging.info("Logging to %s", log_path)
    return logging.getLogger("pseudobulk_runner")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def dense_mean(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
    return np.asarray(matrix, dtype=np.float64).mean(axis=0)


def dense_rows(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def expression_metrics(pred_delta: np.ndarray, true_delta: np.ndarray) -> dict:
    pred_delta = np.asarray(pred_delta, dtype=np.float64).ravel()
    true_delta = np.asarray(true_delta, dtype=np.float64).ravel()
    top = np.argsort(-np.abs(true_delta))[: min(100, true_delta.size)]
    return {
        "pearson_all": pearson_safe(pred_delta, true_delta),
        "mse": float(np.mean((pred_delta - true_delta) ** 2)),
        "pearson_top100": pearson_safe(pred_delta[top], true_delta[top]),
        "sign_agreement": float(np.mean(np.sign(pred_delta) == np.sign(true_delta))),
    }


def _last_finite_scalar(value):
    """Return the last finite scalar from a scvi/scGen history value."""
    if isinstance(value, pd.DataFrame):
        values = value.to_numpy().ravel()
    elif isinstance(value, pd.Series):
        values = value.to_numpy().ravel()
    else:
        try:
            values = np.asarray(value).ravel()
        except Exception:
            return None
    for item in values[::-1]:
        try:
            scalar = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            return scalar
    return None


def extract_training_metrics(model, train_result=None) -> dict:
    """Extract available end-of-training metrics without fabricating values.

    scvi-tools versions expose history slightly differently. Prefer
    ``model.history`` and fall back to the return value from ``model.train``.
    Only metrics that are actually present and finite are returned.
    """
    history = getattr(model, "history", None)
    if not history and train_result is not None:
        history = train_result
    if not isinstance(history, dict):
        return {}

    aliases = {
        "train_loss": ("train_loss", "train_loss_epoch", "loss_train"),
        "validation_loss": ("validation_loss", "validation_loss_epoch", "loss_validation"),
        "elbo_train": ("elbo_train",),
        "elbo_validation": ("elbo_validation",),
        "reconstruction_loss_train": ("reconstruction_loss_train",),
        "reconstruction_loss_validation": ("reconstruction_loss_validation",),
        "kl_local_train": ("kl_local_train",),
        "kl_local_validation": ("kl_local_validation",),
    }
    extracted = {}
    for output_name, candidates in aliases.items():
        for key in candidates:
            if key in history:
                value = _last_finite_scalar(history[key])
                if value is not None:
                    extracted[output_name] = value
                    break
    return extracted



def _history_series(value):
    """Convert a scvi/scGen history value to a 1-D numeric Series."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return pd.Series(dtype=float)
        numeric = value.select_dtypes(include=[np.number])
        if numeric.empty:
            return pd.Series(dtype=float)
        # scvi-tools normally stores one metric column per history entry.
        values = numeric.iloc[:, -1]
    elif isinstance(value, pd.Series):
        values = value
    else:
        try:
            arr = np.asarray(value).reshape(-1)
        except Exception:
            return pd.Series(dtype=float)
        values = pd.Series(arr)
    return pd.to_numeric(values, errors="coerce").reset_index(drop=True)


def extract_training_history(model, train_result=None) -> dict:
    """Return all available epoch-level training metrics, not just final values."""
    history = getattr(model, "history", None)
    if not isinstance(history, dict) or not history:
        history = train_result if isinstance(train_result, dict) else {}
    aliases = {
        "train_loss": ("train_loss", "train_loss_epoch", "loss_train"),
        "validation_loss": ("validation_loss", "validation_loss_epoch", "loss_validation"),
        "elbo_train": ("elbo_train",),
        "elbo_validation": ("elbo_validation",),
        "reconstruction_loss_train": ("reconstruction_loss_train",),
        "reconstruction_loss_validation": ("reconstruction_loss_validation",),
        "kl_global_train": ("kl_global_train",),
        "kl_global_validation": ("kl_global_validation",),
        "kl_local_train": ("kl_local_train",),
        "kl_local_validation": ("kl_local_validation",),
    }
    output = {}
    for output_name, candidates in aliases.items():
        for key in candidates:
            if key in history:
                series = _history_series(history[key])
                if series.notna().any():
                    output[output_name] = series
                    break
    return output

def import_scgen():
    """Import the same external scGen package/API used by scgen_baseline.py."""
    try:
        import scgen
        from scgen._scgenvae import SCGENVAE
    except ImportError as exc:
        raise RuntimeError(
            "Could not import scgen. Install the same GitHub version used by "
            "scgen_baseline.py: pip install 'git+https://github.com/theislab/scgen.git'"
        ) from exc

    # Compatibility patch already present in scgen_baseline.py: newer scvi-tools
    # expects qzm/qzv, while this scGen version emits qz_m/qz_v.
    if not getattr(SCGENVAE.inference, "_cytox_compat_patched", False):
        original_inference = SCGENVAE.inference

        def inference_fixed(self, x):
            output = original_inference(self, x)
            output.setdefault("qzm", output["qz_m"])
            output.setdefault("qzv", output["qz_v"])
            return output

        inference_fixed._cytox_compat_patched = True
        SCGENVAE.inference = inference_fixed
    return scgen


def get_group_profile(adata: ad.AnnData, cell_type: str, cytokine: str,
                      donor: Optional[str] = None) -> np.ndarray:
    mask = (adata.obs["cell_type"].astype(str).to_numpy() == cell_type) & (
        adata.obs["cytokine"].astype(str).to_numpy() == cytokine
    )
    if donor is not None:
        mask &= adata.obs["donor"].astype(str).to_numpy() == donor
    if not np.any(mask):
        raise ValueError(f"No profile for cell_type={cell_type}, cytokine={cytokine}, donor={donor}")
    return dense_mean(adata.X[mask])


def evaluate_prediction(pb: ad.AnnData, pred: ad.AnnData, cytokine: str,
                        holdout: str, strategy: str) -> tuple[dict, dict]:
    """Evaluate predictions against held-out observed treatment responses.

    Donor-specific strategies compare the mean donor-level observed response
    with the model's mean predicted response. If scGen preserves donor metadata
    on its prediction AnnData, donor-specific prediction metrics are also
    calculated and averaged across matched donors.
    """
    obs = pb.obs
    hold = obs[obs["cell_type"].astype(str) == holdout]
    if hold.empty:
        raise ValueError(f"No held-out pseudobulk profiles for {holdout}")

    ctrl = pb[(pb.obs["cell_type"].astype(str) == holdout) &
              (pb.obs["cytokine"].astype(str) == "PBS")]
    stim = pb[(pb.obs["cell_type"].astype(str) == holdout) &
              (pb.obs["cytokine"].astype(str) == cytokine)]
    if ctrl.n_obs == 0 or stim.n_obs == 0:
        raise ValueError(f"Held-out type {holdout} lacks PBS or {cytokine} profiles")

    pooled = (strategy == "pooled_donor")
    if pooled:
        ctrl_mean = dense_mean(ctrl.X)
        true_mean = dense_mean(stim.X)
        true_delta = true_mean - ctrl_mean
        pred_mean = dense_mean(pred.X)
        pred_delta = pred_mean - ctrl_mean
        metrics = expression_metrics(pred_delta, true_delta)
        metrics.update({"n_eval_donors": int(hold["n_donors"].max()),
                        "evaluation_mode": "pooled_population"})
        return metrics, {"control_mean": ctrl_mean, "observed_mean": true_mean,
                         "predicted_mean": pred_mean, "true_delta": true_delta,
                         "pred_delta": pred_delta}

    # Matched donor-specific observed responses; each donor contributes equally.
    donors = sorted(set(ctrl.obs["donor"].astype(str)) & set(stim.obs["donor"].astype(str)))
    if not donors:
        raise ValueError("No matched held-out donors after pseudobulk filtering")
    donor_ctrl, donor_true_delta = {}, {}
    for donor in donors:
        c = ctrl[ctrl.obs["donor"].astype(str) == donor]
        s = stim[stim.obs["donor"].astype(str) == donor]
        donor_ctrl[donor] = dense_mean(c.X)
        donor_true_delta[donor] = dense_mean(s.X) - donor_ctrl[donor]
    ctrl_mean = np.mean(list(donor_ctrl.values()), axis=0)
    true_delta = np.mean(list(donor_true_delta.values()), axis=0)

    pred_obs = pred.obs.copy()
    if "donor" in pred_obs.columns and set(pred_obs["donor"].astype(str)) & set(donors):
        donor_pred_delta = {}
        for donor in donors:
            p = pred[pred.obs["donor"].astype(str) == donor]
            if p.n_obs:
                donor_pred_delta[donor] = dense_mean(p.X) - donor_ctrl[donor]
        common = sorted(set(donor_pred_delta) & set(donor_true_delta))
        if common:
            donor_metrics = [expression_metrics(donor_pred_delta[d], donor_true_delta[d])
                             for d in common]
            metrics = {key: float(np.nanmean([m[key] for m in donor_metrics]))
                       for key in ("pearson_all", "mse", "pearson_top100", "sign_agreement")}
            metrics["evaluation_mode"] = "donor_matched_mean_metrics"
            metrics["n_eval_donors"] = len(common)
            pred_delta = np.mean([donor_pred_delta[d] for d in common], axis=0)
            pred_mean = ctrl_mean + pred_delta
        else:
            pred_mean = dense_mean(pred.X)
            pred_delta = pred_mean - ctrl_mean
            metrics = expression_metrics(pred_delta, true_delta)
            metrics.update({"evaluation_mode": "mean_profile_vs_mean_donor_response",
                            "n_eval_donors": len(donors)})
    else:
        pred_mean = dense_mean(pred.X)
        pred_delta = pred_mean - ctrl_mean
        metrics = expression_metrics(pred_delta, true_delta)
        metrics.update({"evaluation_mode": "mean_profile_vs_mean_donor_response",
                        "n_eval_donors": len(donors)})

    return metrics, {"control_mean": ctrl_mean, "observed_mean": ctrl_mean + true_delta,
                     "predicted_mean": pred_mean, "true_delta": true_delta,
                     "pred_delta": pred_delta,
                     "donor_true_delta": donor_true_delta}


def append_result(path: Path, row: dict):
    """Append a row while tolerating result rows with different optional fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
    fields = old_fields + [key for key in row if key not in old_fields]
    if fields != old_fields:
        # Rewrite only when a later row introduces new columns; this avoids
        # DictWriter's default extrasaction='raise' for skipped/failed runs.
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row)
        os.replace(temp_path, path)
    else:
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=old_fields)
            writer.writerow(row)


def init_wandb(config: dict, run_name: str, args, tags=None):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with 'pip install wandb' or pass --no-wandb."
        ) from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=run_name,
        config=config,
        tags=tags or [],
        reinit="finish_previous",
    )
    return run


try:  # newer scvi-tools uses the `lightning` package, older ones `pytorch_lightning`
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover
    from pytorch_lightning.callbacks import Callback

# Lightning metric names -> names used in W&B.
CURVE_RENAME = {
    "train_loss_epoch": "train_loss",
}
# Only these per-epoch curves are logged (they populate the default Charts).
CURVE_METRICS = (
    "train_loss", "validation_loss",
    "reconstruction_loss_train", "reconstruction_loss_validation",
    "elbo_train", "elbo_validation",
    "kl_local_train", "kl_local_validation",
    "kl_global_train", "kl_global_validation",
)


class WandbEpochLogger(Callback):
    """Log one point per training epoch, live, straight from Lightning.

    Works with run=None (W&B disabled): it still records the last epoch's
    values in ``self.last`` so they can be written to metrics.json.
    """

    def __init__(self, run, logger: logging.Logger):
        # Prefixed names so they can never collide with Lightning's own
        # Callback attributes/methods (newer versions define Callback.log).
        self._wb_run = run
        self._py_logger = logger
        self.n_logged = 0
        self.last: dict = {}
        self._printed_keys = False

    def on_train_epoch_end(self, trainer, pl_module):
        values = {}
        for key, val in trainer.callback_metrics.items():
            if key.endswith("_step"):  # skip per-step duplicates
                continue
            name = CURVE_RENAME.get(key, key)
            if name not in CURVE_METRICS:
                continue
            try:
                values[name] = float(val)
            except (TypeError, ValueError):
                continue
        if not self._printed_keys:
            self._py_logger.info("callback_metrics keys at first epoch end: %s",
                          sorted(trainer.callback_metrics.keys()))
            self._printed_keys = True
        if not values:
            return
        self.last = values
        self.n_logged += 1
        if self._wb_run is not None:
            # Plain metric names + step=epoch -> lands in the default "Charts"
            # section with the same names as earlier runs.
            self._wb_run.log(values, step=trainer.current_epoch)


def run_one(pb: ad.AnnData, cytokine: str, holdout: str, strategy: str,
            min_cells: int, args, logger: logging.Logger) -> dict:
    scgen = import_scgen()
    obs = pb.obs
    cell_types = sorted(obs["cell_type"].astype(str).unique())
    if holdout not in cell_types:
        raise ValueError(f"Holdout '{holdout}' is absent from this pseudobulk dataset")

    # Keep the exact held-out-cell-type split used in scgen_baseline.py:
    # holdout controls remain in training, but holdout stimulated profiles do not.
    is_holdout = obs["cell_type"].astype(str).to_numpy() == holdout
    is_stim = obs["condition"].astype(str).to_numpy() == "stimulated"
    train_mask = ~(is_holdout & is_stim)
    train = pb[train_mask].copy()
    heldout_stim_n = int(np.sum(is_holdout & is_stim))

    if train.n_obs < 8:
        raise ValueError(f"Only {train.n_obs} training profiles; too few for a meaningful VAE run")
    if train.obs["condition"].nunique() < 2:
        raise ValueError("Training data contain fewer than two conditions")

    # Seed scvi-tools as well as NumPy so VAE initialization/sampling is repeatable.
    try:
        import scvi
        scvi.settings.seed = args.seed
    except Exception as seed_exc:
        logger.warning("Could not set scvi-tools seed (%s); continuing", seed_exc)

    # Batch size must be smaller than the training set for tiny pooled runs.
    batch_size = min(args.batch_size, max(2, train.n_obs - 1))
    n_donors = (int(pb.obs.loc[pb.obs["donor"].astype(str) != "POOLED", "donor"].nunique())
                if strategy != "pooled_donor" else int(pb.obs["n_donors"].max()))
    n_train_source_cells = int(train.obs["n_cells"].sum())
    run_config = {
        # Retain the familiar cell-level config names where they have a
        # meaningful pseudobulk equivalent.
        "batch_size": batch_size,
        "cytokine": cytokine,
        "early_stopping_monitor": "validation_loss",
        "holdout": holdout,
        "max_epochs": args.epochs,
        "n_genes": pb.n_vars,
        # Pseudobulk-specific metadata.
        "aggregation_strategy": strategy,
        "min_cells": min_cells,
        "n_donors": n_donors,
        "n_profiles": pb.n_obs,
        "n_train_profiles": train.n_obs,
        "n_heldout_stim_profiles": heldout_stim_n,
        "n_train_source_cells": n_train_source_cells,
        "target_sum": args.target_sum,
        "seed": args.seed,
    }
    # Keep the established cytokine | holdout naming convention and append
    # strategy/threshold so configurations remain distinguishable.
    run_name = f"{cytokine} | {holdout} | {safe_name(strategy)}_n{min_cells}"

    # W&B run is created BEFORE training so curves are logged live, per epoch.
    wandb_run = init_wandb(
        run_config, run_name, args,
        tags=["pseudobulk", strategy, f"min_cells_{min_cells}"],
    )
    epoch_logger = WandbEpochLogger(wandb_run, logger)
    try:
        logger.info("[4/6] Training scGen: %d profiles, batch_size=%d, epochs=%d",
                    train.n_obs, batch_size, args.epochs)
        scgen.SCGEN.setup_anndata(train, batch_key="condition", labels_key="cell_type")
        model = scgen.SCGEN(train)
        train_result = model.train(max_epochs=args.epochs, batch_size=batch_size,
                                   early_stopping=True,
                                   early_stopping_patience=args.patience,
                                   callbacks=[epoch_logger])
        logger.info("Logged %d epoch(s) of training curves", epoch_logger.n_logged)

        # model.history is unreliable on newer scvi-tools; keep it only as a
        # fallback when the live callback captured nothing.
        training_history = extract_training_history(model, train_result)
        if wandb_run is not None and epoch_logger.n_logged == 0 and training_history:
            logger.warning("Callback logged nothing; replaying model.history instead")
            n_hist = max(len(series) for series in training_history.values())
            for epoch in range(n_hist):
                row = {key: float(series.iloc[epoch])
                       for key, series in training_history.items()
                       if epoch < len(series) and pd.notna(series.iloc[epoch])}
                if row:
                    wandb_run.log(row, step=epoch)
        training_metrics = {
            key: value.dropna().iloc[-1].item()
            for key, value in training_history.items() if value.notna().any()
        }
        if not training_metrics and epoch_logger.last:
            training_metrics = dict(epoch_logger.last)
        if epoch_logger.n_logged == 0 and not training_history:
            logger.warning("No epoch-level training metrics were captured")

        logger.info("[5/6] Predicting held-out stimulated profiles for %s", holdout)
        pred, _ = model.predict(ctrl_key="control", stim_key="stimulated",
                                celltype_to_predict=holdout)
        if pred is None or pred.n_obs == 0:
            raise RuntimeError("scGen returned an empty prediction")

        metrics, arrays = evaluate_prediction(pb, pred, cytokine, holdout, strategy)

        logger.info("[6/6] Results | %s | holdout=%s | strategy=%s | min_cells=%d",
                    cytokine, holdout, strategy, min_cells)
        logger.info("scGen: r=%.4f mse=%.6f top100_r=%.4f sign=%.3f | mode=%s | n_eval_donors=%s",
                    metrics["pearson_all"], metrics["mse"], metrics["pearson_top100"],
                    metrics["sign_agreement"], metrics["evaluation_mode"], metrics["n_eval_donors"])

        if wandb_run is not None:
            # One final value per run -> summary, not history.
            wandb_run.summary.update({
                "scGen/r_all": metrics["pearson_all"],
                "scGen/r_top100": metrics["pearson_top100"],
                "scGen/mse": metrics["mse"],
                "scGen/sign_agreement": metrics["sign_agreement"],
                "data/n_profiles": pb.n_obs,
                "data/n_train_profiles": train.n_obs,
                "data/n_heldout_stim_profiles": heldout_stim_n,
                "data/n_eval_donors": metrics["n_eval_donors"],
                "data/n_donors": n_donors,
                "data/n_train_source_cells": n_train_source_cells,
                "data/median_cells_per_pseudobulk": float(pb.obs["n_cells"].median()),
                "data/min_cells_observed": int(pb.obs["n_cells"].min()),
                "data/max_cells_observed": int(pb.obs["n_cells"].max()),
            })
    finally:
        # Always close the run, even if predict/evaluate raised.
        if wandb_run is not None:
            wandb_run.finish()

    tag = "__".join([safe_name(strategy), f"min{min_cells}",
                     safe_name(cytokine), safe_name(holdout)])
    run_dir = Path(args.outdir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(run_dir / "prediction.h5ad", compression="gzip")
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump({"config": run_config, "scgen": metrics,
                   "training_metrics": training_metrics},
                  handle, indent=2, default=str)
    if args.save_model:
        model.save(str(run_dir / "model"), overwrite=True)

    result = {
        **run_config,
        "evaluation_mode": metrics["evaluation_mode"],
        "n_eval_donors": metrics["n_eval_donors"],
        **{f"scgen_{k}": v for k, v in metrics.items()
           if k not in {"evaluation_mode", "n_eval_donors"}},
        **{f"training_{k}": v for k, v in training_metrics.items()},
        "run_name": run_name,
        "status": "success",
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare donor-aware HCD pseudobulk strategies with scGen."
    )
    parser.add_argument("--data", default="data/hcd_subset.h5ad")
    parser.add_argument("--cytokines", nargs="+", default=None,
                        help="Cytokines to test; defaults to all non-PBS conditions in the file.")
    parser.add_argument("--holdouts", nargs="+", default=None,
                        help="Held-out cell types; defaults to all non-rare cell types.")
    parser.add_argument("--strategies", nargs="+", choices=DEFAULT_STRATEGIES,
                        default=DEFAULT_STRATEGIES)
    parser.add_argument("--thresholds", nargs="+", type=int, default=DEFAULT_THRESHOLDS,
                        help="Minimum cells for donor-specific sum_counts.")
    parser.add_argument("--mean-log-min-cells", type=int, default=10,
                        help="Minimum cells for the mean_log comparison (default: 10).")
    parser.add_argument("--pooled-min-cells", type=int, default=10,
                        help="Minimum pooled cells per cell-type/condition for pooled_donor.")
    parser.add_argument("--min-eval-donors", type=int, default=3,
                        help="Skip donor-specific runs with fewer matched evaluation donors.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-rare", action="store_true")
    parser.add_argument("--save-model", action="store_true",
                        help="Save each trained scGen model (uses additional disk space).")
    parser.add_argument("--outdir", default="data/pseudobulk_runs")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=WANDB_MODE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and summarize pseudobulk datasets without training scGen.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    logger = configure_logging(outdir, args.log_level)
    np.random.seed(args.seed)

    logger.info("=" * 68)
    logger.info("HCD PSEUDOBULK EXPERIMENT RUNNER")
    logger.info("=" * 68)
    logger.info("Input: %s", args.data)
    logger.info("Strategies: %s", args.strategies)
    logger.info("Donor-specific thresholds: %s", args.thresholds)
    logger.info("Pooled minimum cells: %d", args.pooled_min_cells)
    logger.info("W&B: %s", "disabled" if args.no_wandb else
                f"{args.wandb_project} ({args.wandb_mode})")

    if not Path(args.data).exists():
        raise FileNotFoundError(f"HCD subset not found: {args.data}")
    logger.info("[1/6] Loading HCD subset...")
    adata = ad.read_h5ad(args.data)
    logger.info("Loaded %s cells x %s genes", f"{adata.n_obs:,}", f"{adata.n_vars:,}")
    for key in ["donor", "cell_type", "cytokine"]:
        if key not in adata.obs:
            raise KeyError(f"Required metadata column '{key}' missing from adata.obs")
    if "counts" not in adata.layers:
        raise KeyError("adata.layers['counts'] is required for count-based pseudobulking")
    logger.info("Donors=%d | cell types=%d | conditions=%s",
                adata.obs["donor"].nunique(), adata.obs["cell_type"].nunique(),
                sorted(adata.obs["cytokine"].astype(str).unique()))

    if not args.keep_rare:
        before = adata.n_obs
        adata = adata[~adata.obs["cell_type"].astype(str).isin(RARE_CELL_TYPES)].copy()
        logger.info("Excluded rare cell types %s: %s -> %s cells",
                    RARE_CELL_TYPES, f"{before:,}", f"{adata.n_obs:,}")

    available_cytokines = sorted(set(adata.obs["cytokine"].astype(str)) - {"PBS"})
    cytokines = args.cytokines or available_cytokines
    missing_cytokines = set(cytokines) - set(available_cytokines)
    if missing_cytokines:
        raise ValueError(f"Requested cytokines absent from data: {sorted(missing_cytokines)}")
    available_holdouts = sorted(adata.obs["cell_type"].astype(str).unique())
    holdouts = args.holdouts or available_holdouts
    missing_holdouts = set(holdouts) - set(available_holdouts)
    if missing_holdouts:
        raise ValueError(f"Requested holdouts absent from data: {sorted(missing_holdouts)}")

    configs = []
    for strategy in args.strategies:
        if strategy == "pooled_donor":
            thresholds = [args.pooled_min_cells]
        elif strategy == "mean_log":
            thresholds = [args.mean_log_min_cells]
        else:
            thresholds = args.thresholds
        for threshold in thresholds:
            configs.append((strategy, threshold))
    estimated_runs = len(configs) * len(cytokines) * len(holdouts)
    logger.info("Planned configurations: %s", configs)
    logger.info("Cytokines=%d | holdouts=%d | estimated model runs=%d",
                len(cytokines), len(holdouts), estimated_runs)

    results_path = outdir / "results.csv"
    summary_path = outdir / "configuration_summary.csv"
    completed = 0
    failures = 0
    for cytokine in cytokines:
        for strategy, threshold in configs:
            logger.info("\n" + "#" * 68)
            logger.info("[2/6] Pseudobulk construction: %s | %s | min_cells=%d",
                        cytokine, strategy, threshold)
            try:
                pb = build_pseudobulk(
                    adata, cytokine=cytokine, strategy=strategy,
                    min_cells=threshold, target_sum=args.target_sum,
                )
            except Exception:
                failures += len(holdouts)
                logger.exception("Could not construct pseudobulk for %s/%s/min%d",
                                 cytokine, strategy, threshold)
                continue

            summary = {
                "cytokine": cytokine, "strategy": strategy, "min_cells": threshold,
                "n_profiles": pb.n_obs, "n_genes": pb.n_vars,
                "n_cells_aggregated": int(pb.obs["n_cells"].sum()),
                "min_cells_observed": int(pb.obs["n_cells"].min()),
                "median_cells_observed": float(pb.obs["n_cells"].median()),
                "max_cells_observed": int(pb.obs["n_cells"].max()),
                "n_cell_types": pb.obs["cell_type"].nunique(),
                "n_donors": int(pb.obs.loc[pb.obs["donor"] != "POOLED", "donor"].nunique())
                            if strategy != "pooled_donor" else
                            int(pb.obs["n_donors"].max()),
            }
            logger.info("[3/6] Dataset summary: %s", summary)
            append_result(summary_path, summary)
            pb_dir = outdir / "pseudobulk_data" / (
                f"{safe_name(strategy)}_min{threshold}_{safe_name(cytokine)}"
            )
            pb_dir.mkdir(parents=True, exist_ok=True)
            pb_path = pb_dir / "profiles.h5ad"
            if not pb_path.exists():
                pb.write_h5ad(pb_path, compression="gzip")
                logger.info("Saved pseudobulk matrix: %s", pb_path)
            if args.dry_run:
                logger.info("Dry run: skipping model training")
                continue

            for holdout in holdouts:
                logger.info("\nRun %d/%d: %s | %s | holdout=%s | min_cells=%d",
                            completed + failures + 1, estimated_runs,
                            strategy, cytokine, holdout, threshold)
                donor_summary = paired_donor_summary(pb, cytokine, holdout)
                if strategy != "pooled_donor":
                    n_eval_donors = (int(donor_summary.loc[
                        donor_summary["cell_type"] == holdout, "n_paired_donors"].max())
                        if not donor_summary.empty and
                        np.any(donor_summary["cell_type"] == holdout) else 0)
                    logger.info("Held-out donor pairs available: %d", n_eval_donors)
                    if n_eval_donors < args.min_eval_donors:
                        failures += 1
                        logger.warning("Skipping: %d matched donors < minimum %d",
                                       n_eval_donors, args.min_eval_donors)
                        append_result(results_path, {
                            "cytokine": cytokine, "holdout": holdout,
                            "strategy": strategy, "min_cells": threshold,
                            "status": "skipped_insufficient_eval_donors",
                            "n_eval_donors": n_eval_donors,
                        })
                        continue
                try:
                    result = run_one(pb, cytokine, holdout, strategy, threshold, args, logger)
                    append_result(results_path, result)
                    completed += 1
                except Exception as exc:
                    failures += 1
                    logger.error("FAILED %s | %s | %s | min%d: %s",
                                 strategy, cytokine, holdout, threshold, exc)
                    logger.debug(traceback.format_exc())
                    append_result(results_path, {
                        "cytokine": cytokine, "holdout": holdout,
                        "strategy": strategy, "min_cells": threshold,
                        "status": "failed", "error": str(exc)[:500],
                    })

    logger.info("=" * 68)
    logger.info("EXPERIMENTS FINISHED: completed=%d failures/skips=%d", completed, failures)
    logger.info("Results: %s", results_path)
    logger.info("Logs: %s", outdir / "logs")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()