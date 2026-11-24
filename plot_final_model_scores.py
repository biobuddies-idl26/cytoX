#!/usr/bin/env python3
"""Plot the final eight-model comparison from the similarity-transfer ablation.

No H5AD, scGen, PyTorch, GPU, or model training is needed. Reads the final
results.csv (or the output directory / ZIP containing it). All eight models
are compared using real PBS; the proposed model is fixed to K=2, equal weights.
An optional reconstructed-PBS supplement compares only the six VAE models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

MODELS = [
    ('original_naive', 0, 'all_equal', 'Original naive', '#9AA2AC'),
    ('strong_naive_K1', 1, 'equal', 'Strong naive K1', '#C18A31'),
    ('original_scgen_Z100', 0, 'none', 'Original scGen Z100', '#6B7F98'),
    ('shared_scgen_Z100', 0, 'none', 'Shared scGen Z100', '#6B7F98'),
    ('shared_scgen_Z200', 0, 'none', 'Shared scGen Z200', '#6B7F98'),
    ('matched_all_latent_Z200', 0, 'all_equal', 'Shared Z200 | all sources', '#6B7F98'),
    ('shared_latent_transfer', 2, 'proportional', 'Shared Z200 | K2 proportional', '#5E9CBB'),
    ('shared_latent_transfer', 2, 'equal', 'Shared Z200 | K2 equal (proposed)', '#007F73'),
]
METRICS = [('r_all', 'Pearson r | all genes', True, '.4f'),
           ('r_top100', 'Pearson r | top 100 genes', True, '.4f'),
           ('mse', 'Mean squared error', False, '.6f')]
CYTOKINES = ['IFN-beta', 'IFN-gamma', 'IL-1-beta', 'IL-15', 'IL-32-beta', 'IL-4']


def read_results(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / 'results.csv'
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            entries = [n for n in z.namelist() if Path(n).name == 'results.csv' and not n.startswith('__MACOSX/')]
            if len(entries) != 1:
                raise ValueError(f'Expected exactly one results.csv in {path}; found {entries}')
            with z.open(entries[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def select_models(df, holdout, reference):
    required = {'holdout','family','k','weighting','reference','cytokine','r_all','r_top100','mse'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Not the final ablation results.csv. Missing columns: {sorted(missing)}')
    d = df.loc[(df.holdout == holdout) & (df.reference == reference)].copy()
    d['k'] = pd.to_numeric(d['k'], errors='raise')
    specs = MODELS if reference == 'real_PBS' else MODELS[2:]
    chunks = []
    for family,k,weight,label,color in specs:
        v = d.loc[(d.family == family) & (d.k == k) & (d.weighting == weight)].copy()
        if v.cytokine.duplicated().any() or set(v.cytokine) != set(CYTOKINES):
            raise ValueError(f'{label}, {reference}: expected exactly one row per each of the six cytokines; got {v.cytokine.tolist()}. Use the final v2 ablation output.')
        for metric,_,_,_ in METRICS:
            v[metric] = pd.to_numeric(v[metric], errors='raise')
            if not np.isfinite(v[metric]).all():
                raise ValueError(f'{label}: non-finite {metric}')
            if metric == 'mse' and (v[metric] < 0).any():
                raise ValueError(f'{label}: negative MSE')
            if metric != 'mse' and (v[metric].abs() > 1+1e-8).any():
                raise ValueError(f'{label}: correlation outside [-1,1]')
        v['model'] = label
        chunks.append(v)
    selected = pd.concat(chunks, ignore_index=True)
    labels = [s[3] for s in specs]
    selected['model'] = pd.Categorical(selected.model, labels, ordered=True)
    selected['cytokine'] = pd.Categorical(selected.cytokine, CYTOKINES, ordered=True)
    return selected.sort_values(['model','cytokine']).reset_index(drop=True), specs


def summarize(d, specs):
    rows = []
    for *_, label, color in specs:
        a = d[d.model == label]
        r = {'holdout': str(a.holdout.iloc[0]), 'reference': str(a.reference.iloc[0]),
             'model': label, 'n_cytokines': len(a)}
        for metric,_,higher,_ in METRICS:
            r[metric] = a[metric].mean()
            r[metric+'_sd'] = a[metric].std(ddof=1)
            # Number of cytokines for which this model is best among the displayed models.
            p = d.pivot(index='model', columns='cytokine', values=metric)
            best = p.max(axis=0) if higher else p.min(axis=0)
            r[metric+'_n_best'] = int(np.isclose(p.loc[label], best, rtol=0, atol=1e-12).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
                         'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':0.8,'xtick.labelsize':10,'ytick.labelsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'savefig.facecolor':'white','figure.facecolor':'white'})


def save(fig, folder, name, dpi):
    for ext in ('png','pdf','svg'):
        fig.savefig(folder / f'{name}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=0.13)
    plt.close(fig)


def overview(d, summary, specs, reference, folder, dpi):
    labels = [s[3] for s in specs]
    colors = [s[4] for s in specs]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1,3,figsize=(16.2,5.6), sharey=True)
    fig.subplots_adjust(left=.235,right=.96,top=.77,bottom=.20,wspace=.46)
    fig.suptitle('Cytokine-response prediction: final model comparison', fontsize=19, fontweight='bold', y=.98)
    ref = 'real PBS' if reference == 'real_PBS' else 'reconstructed PBS (VAE models only)'
    fig.text(.5,.90,f'{d.holdout.iloc[0]} held out | {ref} | six cytokines | fixed K = 2 for proposed model',ha='center',color='#4A596B',fontsize=11)
    for j,(ax,(m,title,higher,fmt)) in enumerate(zip(axes,METRICS)):
        means=summary[m].to_numpy()
        vals=d[m].to_numpy()
        ax.barh(y,means,height=.56,color=colors,alpha=.85,zorder=2)
        # All cytokine scores shown as dots, not confidence intervals.
        for i,label in enumerate(labels):
            x=d.loc[d.model==label,m].to_numpy()
            jitter=np.linspace(-.18,.18,len(x))
            ax.scatter(x,y[i]+jitter,s=17,facecolors='white',edgecolors='#334155',linewidths=.65,zorder=4)
            ax.text(1.025,y[i],format(means[i],fmt),transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9.2, fontweight='bold' if np.isclose(means[i], means.max() if higher else means.min(),rtol=0,atol=1e-12) else 'normal',clip_on=False)
        ax.set_title(f'{chr(65+j)}  {title}',loc='left',fontweight='bold',pad=12)
        ax.set_xlabel('Higher is better' if higher else 'Lower is better')
        ax.set_xlim(min(0, vals.min()*1.05),1.0 if higher else max(vals.max()*1.06, .001))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='x',color='#DCE2E8',linewidth=.7,zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks(y,labels)
        ax.tick_params(axis='y',length=0,labelleft=(j==0))
        ax.spines['left'].set_visible(False)
    axes[0].invert_yaxis()
    for label in axes[0].get_yticklabels():
        if 'proposed' in label.get_text(): label.set_fontweight('bold'); label.set_color('#007F73')
    fig.legend(handles=[Rectangle((0,0),1,1,facecolor='#8398AB',label='Bar: mean across six cytokines'),Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor='#334155',label='Dot: individual cytokine')],loc='lower center',bbox_to_anchor=(.55,.035),ncol=2,frameon=False)
    save(fig,folder,'01_all_models_overall_'+reference,dpi)


def heatmaps(d,specs,reference,folder,dpi):
    labels=[s[3] for s in specs]
    fig,axes=plt.subplots(1,3,figsize=(17.4,5.6),sharey=True)
    fig.subplots_adjust(left=.21,right=.96,top=.76,bottom=.21,wspace=.37)
    fig.suptitle('Prediction scores for individual cytokines', fontsize=19,fontweight='bold',y=.98)
    ref='real PBS' if reference=='real_PBS' else 'reconstructed PBS (VAE models only)'
    fig.text(.5,.90,f'{d.holdout.iloc[0]} held out | {ref} | outlined cells = best score for each cytokine',ha='center',color='#4A596B',fontsize=11)
    for j,(ax,(m,title,higher,fmt)) in enumerate(zip(axes,METRICS)):
        p=d.pivot(index='model',columns='cytokine',values=m).reindex(index=labels,columns=CYTOKINES)
        a=p.to_numpy()
        # Independent scale per metric; higher color intensity always means better performance.
        lo,hi=a.min(),a.max()
        if lo==hi: hi=lo+1e-9
        cmap=plt.get_cmap('YlGnBu' if higher else 'YlGnBu_r')
        norm=Normalize(lo,hi)
        im=ax.imshow(a,aspect='auto',cmap=cmap,norm=norm)
        for i in range(len(labels)):
            for c in range(len(CYTOKINES)):
                v=a[i,c]; rgb=cmap(norm(v))[:3]; luminance=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                ax.text(c,i,f'{v:.3f}' if higher else f'{v:.4f}',ha='center',va='center',fontsize=8.3,color='white' if luminance<.46 else '#17212B')
                best=a[:,c].max() if higher else a[:,c].min()
                if np.isclose(v,best,rtol=0,atol=1e-12):
                    ax.add_patch(Rectangle((c-.48,i-.48),.96,.96,fill=False,edgecolor='#F2A900',linewidth=2.0))
        ax.set_xticks(np.arange(len(CYTOKINES)),CYTOKINES,rotation=40,ha='right',rotation_mode='anchor')
        ax.set_yticks(np.arange(len(labels)),labels)
        ax.tick_params(axis='both',length=0,labelleft=(j==0))
        ax.set_title(f'{chr(65+j)}  {title}',loc='left',fontweight='bold',pad=12)
        ax.spines[:].set_visible(False)
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.26,aspect=25)
        cb.locator=MaxNLocator(4); cb.update_ticks(); cb.ax.tick_params(labelsize=8,length=2)
        cb.outline.set_visible(False)
    for label in axes[0].get_yticklabels():
        if 'proposed' in label.get_text(): label.set_fontweight('bold'); label.set_color('#007F73')
    save(fig,folder,'02_all_models_by_cytokine_'+reference,dpi)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('data/similarity_transfer_FINAL/results.csv'),help='Final results.csv, its directory, or z.zip')
    p.add_argument('--outdir',type=Path,default=Path('data/final_model_comparison_K2'))
    p.add_argument('--holdout',default='CD14 Mono')
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--include-reconstructed',action='store_true',help='Also plot VAE-only reconstructed-PBS supplement; no naive values mixed in')
    args=p.parse_args()
    if args.dpi<72: p.error('--dpi must be >= 72')
    d=read_results(args.results)
    args.outdir.mkdir(parents=True,exist_ok=True)
    style()
    for ref in (['real_PBS','reconstructed_PBS'] if args.include_reconstructed else ['real_PBS']):
        chosen,specs=select_models(d,args.holdout,ref)
        summary=summarize(chosen,specs)
        chosen.to_csv(args.outdir / f'model_scores_by_cytokine_{ref}.csv',index=False,float_format='%.10g')
        summary.to_csv(args.outdir / f'model_scores_summary_{ref}.csv',index=False,float_format='%.10g')
        overview(chosen,summary,specs,ref,args.outdir,args.dpi)
        heatmaps(chosen,specs,ref,args.outdir,args.dpi)
        print('\n'+ref+'\n'+summary[['model','r_all','r_top100','mse']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    (args.outdir/'FIGURE_LEGENDS.txt').write_text(
        'Final model comparison (K2 fixed)\n\n'
        'Figure 01. Horizontal bars represent the unweighted mean over six cytokines; '
        'open points represent individual cytokines (vertical offsets are for visibility only). '
        'Numbers beside bars are means. All eight methods use the same real-PBS reference. '
        'The optional reconstructed-PBS supplement includes only VAE models.\n\n'
        'Figure 02. Each heatmap cell is a single cytokine score. Gold outlines indicate the best '
        'score among plotted models for that cytokine and metric (ties included). Color scales are '
        'metric-specific; darker color indicates better performance.\n\n'
        'Strong naive K1 is learning-free: source selection uses PBS gene-expression cosine similarity. '
        'Proposed shared Z200 K2 uses latent PBS cosine similarity and equal weights. '
        'K2 proportional is retained as the weighting control. Original scGen scores are historical. '
        'Scores are descriptive; K2 was selected during exploratory CD14 Mono analysis, not independently validated.\n',encoding='utf-8')
    print(f'\nSaved figures and tables to {args.outdir.resolve()}')


if __name__=='__main__':
    main()
