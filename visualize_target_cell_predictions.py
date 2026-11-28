#!/usr/bin/env python3
"""Visualize every predicted target cell on the existing target-only UMAP.

Read the cached outputs of visualize_all_latent_shifts.py (hybrid version).
No model loading, retraining, new UMAP fitting, or gene-expression decoding.
Generate one PNG per cytokine: measured response, shared scGen, learned hybrid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

CYTOKINES = ['IFN-beta', 'IFN-gamma', 'IL-1-beta', 'IL-15', 'IL-32-beta', 'IL-4']
MODELS = ('default', 'hybrid')
NAMES = {'default': 'Shared scGen Z200', 'hybrid': 'Learned hybrid'}
COLORS = {'default': '#34699A', 'hybrid': '#8055A1'}
PBS = '#89939F'
OBS = '#222A33'
FILES = ('target_only_umap_coordinates.csv',
         'target_only_umap_predictions.npz',
         'target_only_umap_centroid_endpoints.csv')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path,
                   default=Path('data/all_latent_shifts_Z200_hybrid_v2'),
                   help='Existing visualization run directory, or its Z200 subfolder.')
    p.add_argument('--outdir', type=Path, default=None,
                   help='Output folder. Default: <Z200>/figures/umap/target_only_umap/cell_level_comparison.')
    p.add_argument('--holdout', default='CD14 Mono')
    p.add_argument('--cytokines', nargs='+', default=CYTOKINES)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--cell-arrows', type=int, default=0,
                   help='Optional number of individual PBS-to-prediction arrows per model; 0 hides individual arrows. All predicted cells are always plotted.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--common-limits', action='store_true',
                   help='Use the same axis limits across all cytokines (default: zoom each cytokine, with shared limits across its three panels).')
    a = p.parse_args()
    if a.dpi < 72 or a.cell_arrows < 0 or a.seed < 0:
        p.error('DPI must be >=72; cell-arrows and seed must be nonnegative.')
    if 'PBS' in a.cytokines or len(set(a.cytokines)) != len(a.cytokines):
        p.error('Specify distinct treatment conditions, excluding PBS.')
    return a


def resolve_input(path):
    for folder in (path, path / 'Z200'):
        if all((folder / f).is_file() for f in FILES):
            return folder
    raise FileNotFoundError('Cannot find the three target-only UMAP cache files in '
                            f'{path} or {path / "Z200"}. Run the previous visualization script first.\n'
                            + '\n'.join(FILES))


def finite_xy(x, label):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f'{label}: expected a nonempty finite N x 2 array, got {x.shape}')
    return x


def load_cache(folder, a):
    obs = pd.read_csv(folder / FILES[0], dtype={'cell_id': str, 'cell_type': str,
                                                   'cytokine': str, 'status': str},
                      keep_default_na=False)
    required = {'cell_id', 'cell_type', 'cytokine', 'status', 'UMAP1', 'UMAP2'}
    if not required.issubset(obs.columns):
        raise ValueError(f'Missing columns in measured UMAP coordinates: {required - set(obs.columns)}')
    if set(obs.cell_type) != {a.holdout} or obs.cell_id.duplicated().any():
        raise ValueError('Measured target coordinates have the wrong cell type or duplicate cell IDs.')
    if not obs.loc[obs.cytokine.eq('PBS'), 'status'].eq('reference').all():
        raise ValueError('PBS status must be reference.')
    if not obs.loc[obs.cytokine.ne('PBS'), 'status'].eq('heldout_observed').all():
        raise ValueError('Treated cells must be held-out observed cells.')
    xy = finite_xy(obs[['UMAP1', 'UMAP2']].to_numpy(), 'Measured target coordinates')
    with np.load(folder / FILES[1], allow_pickle=False) as f:
        model = f['model'].astype(str)
        cytokine = f['cytokine'].astype(str)
        pp = np.asarray(f['xy'], dtype=np.float64)
        pbs = finite_xy(f['pbs_xy'], 'Cached PBS coordinates')
        ids = f['cell_id'].astype(str)
    if model.ndim != 1 or cytokine.shape != model.shape or pp.shape != (len(model), len(pbs), 2):
        raise ValueError('Unexpected prediction cache shape.')
    if ids.shape != (len(pbs),) or len(set(ids)) != len(ids) or not np.isfinite(pp).all():
        raise ValueError('Invalid cached prediction coordinates or PBS cell IDs.')
    pbs_rows = obs.loc[obs.cytokine.eq('PBS')].set_index('cell_id')
    if set(pbs_rows.index) != set(ids):
        raise ValueError('Measured and predicted caches contain different PBS cells.')
    measured_pbs = pbs_rows.loc[ids, ['UMAP1', 'UMAP2']].to_numpy(dtype=float)
    if not np.allclose(measured_pbs, pbs, rtol=1e-6, atol=1e-6):
        raise ValueError('PBS coordinates disagree between caches. Do not mix visualization runs.')
    keys = list(zip(model, cytokine))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate model/cytokine entries in prediction cache.')
    predictions = dict(zip(keys, pp))
    ep = pd.read_csv(folder / FILES[2], dtype={'model': str, 'cytokine': str})
    if ep.duplicated(['model', 'cytokine']).any():
        raise ValueError('Duplicate centroid endpoints.')
    ends = {(r.model, r.cytokine): np.array([r.UMAP1, r.UMAP2], dtype=float)
            for r in ep.itertuples(index=False)}
    required_ends = [('PBS', 'PBS')] + [(k, c) for c in a.cytokines for k in ('observed', *MODELS)]
    for key in required_ends:
        if key not in ends or not np.isfinite(ends[key]).all():
            raise ValueError(f'Missing or invalid projected centroid: {key}')
    truth = {}
    for c in a.cytokines:
        truth[c] = finite_xy(xy[obs.cytokine.eq(c).to_numpy()], f'Observed {c}')
        for k in MODELS:
            if (k, c) not in predictions:
                raise ValueError(f'Missing predicted cells for {k}/{c}')
    return pbs, truth, predictions, ends


def limits(*arrays):
    all_xy = np.concatenate([np.asarray(x).reshape(-1, 2) for x in arrays])
    lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
    pad = max(float(np.max(hi - lo)) * .075, .05)
    return ((lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad))


def arrow(ax, start, end, color, ls='-', marker='D'):
    if np.linalg.norm(end - start) > 1e-10:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                     mutation_scale=15, linewidth=2.2,
                                     color=color, linestyle=ls, shrinkA=6,
                                     shrinkB=5, zorder=7))
    ax.scatter(*end, s=85, c=[color], marker=marker, edgecolors='white',
               linewidths=.8, zorder=8)


def draw_panel(ax, pbs, treated, start, observed, bb, pred=None, pred_end=None,
               kind=None, arrow_ids=None):
    # Baseline and observed treatment are identical across all three panels.
    ax.scatter(*pbs.T, s=14, color=PBS, alpha=.30, linewidths=0, zorder=1)
    if pred is None:
        ax.scatter(*treated.T, s=20, color=OBS, alpha=.52, marker='^',
                   linewidths=0, zorder=3)
    else:
        color = COLORS[kind]
        if arrow_ids is not None and len(arrow_ids):
            # These segments connect an untreated cell to ITS OWN computational
            # prediction, never to a measured treated cell.
            segments = np.stack([pbs[arrow_ids], pred[arrow_ids]], axis=1)
            ax.add_collection(LineCollection(segments, colors=[color],
                                             linewidths=.55, alpha=.24, zorder=2))
        ax.scatter(*pred.T, s=19, color=color, alpha=.62, linewidths=0, zorder=3)
        # Open triangles keep the experimentally observed cloud visible even
        # when predicted points overlap it.
        ax.scatter(*treated.T, s=24, facecolors='none', edgecolors=OBS,
                   alpha=.48, marker='^', linewidths=.55, zorder=4)
    arrow(ax, start, observed, OBS, '--', '^')
    if pred is not None:
        arrow(ax, start, pred_end, COLORS[kind], '-', 'D' if kind == 'default' else 'h')
    ax.scatter(*start, s=95, marker='o', facecolors='white', edgecolors='#303C49',
               linewidths=1.6, zorder=10)
    ax.set_xlim(*bb[0]); ax.set_ylim(*bb[1]); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Target UMAP 1'); ax.set_ylabel('Target UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top', 'right']].set_visible(False)


def marker_handle(color, label, marker='o', ls='None', hollow=False):
    return Line2D([], [], label=label, color=color, marker=marker, linestyle=ls,
                  linewidth=1.8, markersize=7, markerfacecolor='none' if hollow else color)


def main():
    a = parse_args()
    folder = resolve_input(a.input_dir)
    out = a.outdir or folder / 'figures' / 'umap' / 'target_only_umap' / 'cell_level_comparison'
    if out.exists() and not out.is_dir():
        raise ValueError(f'Output path is not a directory: {out}')
    pbs, truth, predictions, ends = load_cache(folder, a)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    start = ends['PBS', 'PBS']
    arrow_ids = np.random.default_rng(a.seed).choice(len(pbs), min(a.cell_arrows, len(pbs)), replace=False)
    common_bb = None
    if a.common_limits:
        common_bb = limits(pbs, *truth.values(),
                           *[predictions[k,c] for c in a.cytokines for k in MODELS],
                           *[ends[k,c] for c in a.cytokines for k in ('observed', *MODELS)], start)
    for i, c in enumerate(a.cytokines, 1):
        treated = truth[c]; observed = ends['observed', c]
        bb = common_bb or limits(pbs, treated, predictions['default', c], predictions['hybrid', c],
                                 start, observed, ends['default', c], ends['hybrid', c])
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.6))
        draw_panel(axes[0], pbs, treated, start, observed, bb)
        axes[0].set_title('A  Experimentally observed', fontsize=14, fontweight='bold', pad=13)
        for ax, k, letter in zip(axes[1:], MODELS, ('B', 'C')):
            draw_panel(ax, pbs, treated, start, observed, bb, predictions[k, c],
                       ends[k, c], k, arrow_ids)
            ax.set_title(f'{letter}  {NAMES[k]}', color=COLORS[k], fontsize=14,
                         fontweight='bold', pad=13)
        fig.suptitle(f'{a.holdout} | {c}', fontsize=22, fontweight='bold', y=.99)
        fig.text(.5, .907, f'Individual-cell latent predictions on the same target-only UMAP  |  '
                 f'{len(pbs):,} PBS cells  ·  {len(treated):,} observed treated cells  ·  '
                 f'{len(pbs):,} predictions per model', ha='center', fontsize=11, color='#506074')
        handles = [marker_handle(PBS, 'PBS cells'), marker_handle(OBS, 'Observed treated cells', '^', hollow=True),
                   Line2D([], [], label='PBS centroid', marker='o', ls='', mfc='white', mec='#303C49', mew=1.5, markersize=7),
                   marker_handle(OBS, 'Observed centroid shift', '^', '--'),
                   marker_handle(COLORS['default'], 'Shared scGen predicted cells'),
                   marker_handle(COLORS['hybrid'], 'Hybrid predicted cells'),
                   marker_handle(COLORS['default'], 'Shared scGen centroid shift', 'D', '-'),
                   marker_handle(COLORS['hybrid'], 'Hybrid centroid shift', 'h', '-')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .085), ncol=4,
                   fontsize=9.5, frameon=False, columnspacing=1.35)
        note = ('All PBS cells are projected individually; no pairing with observed treated cells. '
                '\nCentroids are projected from 200D. UMAP overlap is illustrative, not proof of accurate single-cell responses.')
        if a.cell_arrows:
            note = f'{len(arrow_ids)} thin segments per model connect PBS cells to their own predictions.\n' + note
        fig.text(.5, .023, note, ha='center', va='bottom', fontsize=9, color='#506074', wrap=True)
        fig.subplots_adjust(left=.055, right=.985, top=.815, bottom=.225, wspace=.20)
        path = out / f'{i:02d}_{re.sub(r"[^A-Za-z0-9_-]+", "_", c)}.png'
        fig.savefig(path, dpi=a.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved {path}', flush=True)
    print(f'Done: {len(a.cytokines)} PNG figures. Existing UMAP and predictions reused; no models trained.', flush=True)


if __name__ == '__main__':
    main()
