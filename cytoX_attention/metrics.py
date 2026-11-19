#evaluation metrics from the report (section 3.3) + baselines --> numpy / scipy only
import numpy as np
from scipy.stats import mannwhitneyu


def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def top_deg(stim: np.ndarray, ctrl: np.ndarray, n: int = 100) -> np.ndarray:
    #Wilcoxon rank-sum (Mann-Whitney U) per gene, observed treated vs control --> n smallest p-values
    #ties in p broken by |mean difference|
    _, p = mannwhitneyu(stim, ctrl, axis=0)
    p = np.nan_to_num(p, nan=1.0)
    eff = np.abs(stim.mean(0) - ctrl.mean(0))
    return np.lexsort((-eff, p))[:n]


def evaluate(pred_mean: np.ndarray, real_stim: np.ndarray, real_ctrl: np.ndarray, deg_idx=None, n_top: int = 100) -> dict:
    #pred_mean --> predicted mean profile; real_stim / real_ctrl --> observed cells (N, G)
    #R2        --> squared Pearson of predicted vs observed mean expression, all genes
    #r_delta   --> Pearson of predicted vs observed change, top-100 Wilcoxon DEGs (primary metric)
    #mse_delta --> MSE of the change on the same DEGs
    #r_all / r_top100 / mse --> same columns as scgen_baseline.py (top-100 by |observed change|)
    ctrl = real_ctrl.mean(0)
    real = real_stim.mean(0)
    d_pred, d_real = pred_mean - ctrl, real - ctrl
    if deg_idx is None:
        deg_idx = top_deg(real_stim, real_ctrl, n_top)
    top_abs = np.argsort(-np.abs(d_real))[:n_top]
    return {
        "R2": _pearson(pred_mean, real) ** 2,
        "r_delta": _pearson(d_pred[deg_idx], d_real[deg_idx]),
        "mse_delta": float(np.mean((d_pred[deg_idx] - d_real[deg_idx]) ** 2)),
        "r_all": _pearson(d_pred, d_real),
        "mse": float(np.mean((d_pred - d_real) ** 2)),
        "r_top100": _pearson(d_pred[top_abs], d_real[top_abs]),
    }


def naive_shift(X: np.ndarray, cell_type: np.ndarray, cytokine: np.ndarray, holdout: str, c: str, ctrl: str = "PBS") -> np.ndarray:
    #report Eq. 4 --> mean over training types of (mean treated - mean control) added to held-out control mean
    ho_ctrl = X[(cell_type == holdout) & (cytokine == ctrl)].mean(0)
    shifts = []
    for t in np.unique(cell_type):
        if t == holdout:
            continue
        s, k = (cell_type == t) & (cytokine == c), (cell_type == t) & (cytokine == ctrl)
        if s.any() and k.any():
            shifts.append(X[s].mean(0) - X[k].mean(0))
    return ho_ctrl + np.mean(shifts, axis=0)


def unchanged(X: np.ndarray, cell_type: np.ndarray, cytokine: np.ndarray, holdout: str, ctrl: str = "PBS") -> np.ndarray:
    #floor baseline --> predicts no response
    return X[(cell_type == holdout) & (cytokine == ctrl)].mean(0)
