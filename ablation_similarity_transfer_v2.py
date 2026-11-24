#!/usr/bin/env python3
r"""Focused, publication-oriented inference-only ablation for shared scGen Z200.

Run from the repository root:
  CUDA_VISIBLE_DEVICES=0 python -u ablation_similarity_transfer.py \
    --data data/hcd_subset.h5ad --shared-z100 data/shared_CD14_Mono \
    --shared-z200 data/shared_CD14_Mono_Z200 --holdout 'CD14 Mono' \
    --device cuda --outdir data/similarity_transfer_Z200

No VAE training, no donor splitting, no Z400. Gene-space cosine is used
only for the independent, learning-free K1 baseline.
Rank source cell types by cosine similarity of frozen Z200 PBS centroids.
Compare equal and positive-cosine proportional weights for K=1..10.
Matched-size random source subsets are the selection control; all K1/K2
subsets are evaluated, and other K values are sampled reproducibly.
Both naive baselines are VAE-free: all-source equal gene transfer and
nearest-one gene transfer using cosine of mean PBS expression.
All K2 pairs are evaluated for a descriptive similarity-performance scatter
plot and Spearman correlations. No VAE-guided gene-transfer arm.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import warnings
import numpy as np
import pandas as pd

CYTOKINES = ['IFN-beta', 'IFN-gamma', 'IL-1-beta', 'IL-15', 'IL-32-beta', 'IL-4']
RARE = ['pDC', 'HSPC', 'ILC', 'Plasmablast']
METRICS = ['r_all','r_top100','mse']
METRIC_LABEL = {'r_all':'Pearson r | all genes', 'r_top100':'Pearson r | top 100 genes', 'mse':'MSE'}
TOL = {'r_all':0.0000501,'r_top100':0.0000501,'mse':0.000000501}
REFS = ['real_PBS','reconstructed_PBS']
EQUAL='#36639A'; PROP='#007F71'; RAND='#949BA5'; GRAY='#3C4856'; GOLD='#B7791F'
BENCH = {
 'original_naive':'Original naive (all sources)',
 'strong_naive_K1':'Strong naive K1 (gene cosine)',
 'original_scgen_Z100':'Original scGen Z100',
 'shared_scgen_Z100':'Shared scGen Z100',
 'shared_scgen_Z200':'Shared scGen Z200',
 'matched_all_latent_Z200':'Shared Z200 | all sources, matched',
}

def dense(x):
    return x.toarray() if hasattr(x,'toarray') else np.asarray(x)

def score(pred,base,real_delta):
    # Identical to the original repository's delta evaluation.
    d=pred-base
    top=np.argsort(-np.abs(real_delta))[:100]
    with np.errstate(invalid='ignore',divide='ignore'):
        return dict(r_all=float(np.corrcoef(d,real_delta)[0,1]),
                    r_top100=float(np.corrcoef(d[top],real_delta[top])[0,1]),
                    mse=float(np.mean((d-real_delta)**2)))

def matches(a,b):
    return all(np.isclose(float(a[m]),float(b[m]),atol=TOL[m],rtol=0) for m in METRICS)

def read_baselines(path,holdout,cytokines,run=None):
    df=pd.read_csv(path)
    if run is not None: df=df[df['run'].eq(run)]
    methods=['scGen','scGen_vs_real_ctrl','naive']
    df=df[df.holdout.eq(holdout)&df.cytokine.isin(cytokines)&df.model.isin(methods)]
    if set(zip(df.cytokine,df.model))!={(c,m) for c in cytokines for m in methods} or df.duplicated(['cytokine','model']).any():
        raise ValueError(f'Missing or duplicate benchmark rows: {path}')
    if not np.isfinite(df[METRICS].to_numpy(dtype=float)).all(): raise ValueError(f'Nonfinite scores: {path}')
    return df.set_index(['cytokine','model'])

def cosine(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b)
    if np.any(den<=1e-12) or not np.isfinite(den).all(): raise ValueError('Invalid PBS centroid for cosine')
    return np.clip((a@b)/den,-1,1)

def safe_cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/den,-1,1)) if den>1e-12 else np.nan

def weights_for(s,indices,kind):
    idx=np.asarray(indices,dtype=int); w=np.zeros(len(s),dtype=np.float64)
    if kind=='equal': local=np.ones(len(idx)); fallback=False
    elif kind=='proportional':
        local=np.maximum(s[idx],0.); fallback=bool(local.sum()<=1e-12)
        if fallback: local=np.ones(len(idx))
    else: raise ValueError(kind)
    w[idx]=local/local.sum()
    assert np.isfinite(w).all() and np.all(w>=0) and np.isclose(w.sum(),1)
    return w,fallback

def make_design(s,sources,kmax,nrandom,seed):
    # Source names are alphabetical, stable tie break. All decisions use PBS only.
    order=np.argsort(-s,kind='stable'); rank=np.empty(len(s),int); rank[order]=np.arange(1,len(s)+1)
    top=[]; random=[]; random_meta=[]; rng=np.random.default_rng(seed)
    for k in range(1,kmax+1):
        idx=order[:k]
        for wt in ['equal','proportional']:
            w,fb=weights_for(s,idx,wt)
            top.append(dict(k=k,weighting=wt,w=w,indices=tuple(idx),fallback=fb))
        # Unique source subsets, sampled uniformly without replacement, fixed
        # across cytokines. Exact enumeration for K<=2 (and small universes).
        if k<=2 or nrandom>0:
            universe=list(itertools.combinations(range(len(s)),k))
            exhaustive=k<=2 or len(universe)<=nrandom
            if exhaustive: chosen=universe
            else: chosen=[universe[i] for i in sorted(rng.choice(len(universe),size=nrandom,replace=False))]
            for j,idxr in enumerate(chosen,1):
                ident=f'K{k:02d}_S{j:04d}'
                wr,_=weights_for(s,idxr,'equal')
                random.append(dict(k=k,subset_id=ident,w=wr,indices=idxr))
                random_meta.append(dict(k=k,subset_id=ident,source_cell_types=' | '.join(sources[i] for i in idxr),
                    exhaustive=exhaustive,total_possible=len(universe),is_top_k=set(idxr)==set(idx),mean_similarity=float(wr@s)))
    return order,rank,top,random,pd.DataFrame(random_meta)

def decode_mean(model,z,torch,batch_size):
    # Decode individual cells, then average. GPU reduction avoids transferring
    # the entire 500 x 7000 matrix for every random-subset control.
    dev=next(model.module.parameters()).device
    total=None
    with torch.inference_mode():
        for i in range(0,len(z),batch_size):
            x=torch.as_tensor(z[i:i+batch_size],device=dev,dtype=torch.float32)
            v=model.module.generative(x)['px'].sum(dim=0)
            total=v if total is None else total+v
    return (total/len(z)).detach().cpu().numpy()

def save_csv(df,path):
    df.to_csv(path,index=False,float_format='%.10g')

def result(c,holdout,family,ref,metrics,k=0,weighting='none',origin='computed'):
    return dict(cytokine=c,holdout=holdout,family=family,reference=ref,k=k,weighting=weighting,origin=origin,**metrics)

GROUPS=['holdout','family','reference','k','weighting','origin']

def summarize(df):
    ag={'n_cytokines':('cytokine','nunique')}
    for m in METRICS: ag.update({m:(m,'mean'),m+'_sd':(m,'std'),m+'_n':(m,'count')})
    return df.groupby(GROUPS,dropna=False,sort=False).agg(**ag).reset_index()

def paired(left,right,keys,contrast):
    if left.empty or right.empty: return pd.DataFrame()
    d=left[keys+METRICS].merge(right[keys+METRICS],on=keys,suffixes=('_test','_control'),validate='one_to_one')
    d['contrast']=contrast
    for m in METRICS:
        sign=-1 if m=='mse' else 1
        d[m+'_gain']=sign*(d[m+'_test']-d[m+'_control'])
    return d

def gain_summary(d,keys):
    ag={'n_cytokines':('cytokine','nunique')}
    for m in METRICS:
        col=m+'_gain'
        ag.update({col:(col,'mean'),col+'_sd':(col,'std'),col+'_n':(col,'count'),m+'_wins':(col,lambda x:int((x>1e-10).sum()))})
    return d.groupby(keys,dropna=False,sort=False).agg(**ag).reset_index()

def dilution_row(c,holdout,k,wt,w,s,lp,anchor,order,pred,ctrl,recon,real_delta):
    dz=w@lp; norm=np.linalg.norm(dz); refnorm=np.linalg.norm(anchor)
    denom=float(w@np.linalg.norm(lp,axis=1)); gt=np.linalg.norm(real_delta)
    return dict(cytokine=c,holdout=holdout,k=k,weighting=wt,
        top2_weight=float(w[order[:2]].sum()),weighted_source_similarity=float(w@s),
        effective_sources=float(1/np.sum(w*w)),latent_shift_norm=float(norm),
        latent_norm_relative_to_K2=float(norm/refnorm) if refnorm>1e-12 else np.nan,
        latent_cosine_to_K2=safe_cosine(dz,anchor),
        cancellation_index=float(norm/denom) if denom>1e-12 else np.nan,
        decoded_delta_norm_ratio_real=float(np.linalg.norm(pred-ctrl)/gt) if gt>1e-12 else np.nan,
        decoded_delta_norm_ratio_reconstructed=float(np.linalg.norm(pred-recon)/gt) if gt>1e-12 else np.nan)

def selection_summaries(random_results,meta,sweep):
    if random_results.empty: return pd.DataFrame(),pd.DataFrame()
    # A subset is the unit for the random band: average its six cytokine scores
    # first, then compute distribution across distinct source subsets.
    sm=random_results.groupby(['holdout','reference','k','subset_id'],sort=False).agg(
        n_cytokines=('cytokine','nunique'),**{m:(m,'mean') for m in METRICS}).reset_index().merge(meta,on=['k','subset_id'],validate='many_to_one')
    tt=summarize(sweep[sweep.family.eq('shared_latent_transfer')&sweep.weighting.eq('equal')])
    rows=[]
    for (h,ref,k),g in sm.groupby(['holdout','reference','k'],sort=False):
        t=tt[tt.reference.eq(ref)&tt.k.eq(k)&tt.holdout.eq(h)].iloc[0]
        for m in METRICS:
            v=g[m].to_numpy();v=v[np.isfinite(v)]; a=float(t[m]); sign=-1 if m=='mse' else 1
            if not len(v):continue
            rows.append(dict(holdout=h,reference=ref,k=k,metric=m,n_subsets=len(v),total_possible=int(g.total_possible.iloc[0]),
                exhaustive=bool(g.exhaustive.iloc[0]),top_k=a,random_mean=float(v.mean()),random_median=float(np.median(v)),
                random_q10=float(np.quantile(v,.10)),random_q90=float(np.quantile(v,.90)),
                top_minus_random_median=float(sign*(a-np.median(v))),
                fraction_subsets_top_beats=float(np.mean(sign*(a-v)>1e-10)),
                fraction_subsets_tied=float(np.mean(np.abs(a-v)<=1e-10))))
    return sm,pd.DataFrame(rows)


def analyze_pairs(random_means, n_sources, n_cytokines):
    """Descriptive association across all distinct equal-weight K2 pairs."""
    from scipy.stats import spearmanr
    pairs=random_means[random_means.k.eq(2)].copy()
    expected=n_sources*(n_sources-1)//2
    rows=[]
    for (holdout,ref),g in pairs.groupby(['holdout','reference'],sort=False):
        if len(g)!=expected or g.subset_id.duplicated().any() or not g.exhaustive.all():
            raise RuntimeError('K2 pair analysis requires complete enumeration of distinct pairs')
        if not g.n_cytokines.eq(n_cytokines).all() or not np.isfinite(g[METRICS+['mean_similarity']].to_numpy()).all():
            raise RuntimeError('K2 pair analysis requires complete, finite scores for every pair')
        if g.is_top_k.sum()!=1:raise RuntimeError('Expected exactly one similarity-selected K2 pair')
        for m in METRICS:
            rho=float(spearmanr(g.mean_similarity,g[m]).statistic) if g.mean_similarity.nunique()>1 and g[m].nunique()>1 else np.nan
            ranks=g[m].rank(method='min',ascending=(m=='mse')).astype(int)
            pairs.loc[g.index,m+'_rank']=ranks
            rows.append(dict(holdout=holdout,reference=ref,metric=m,n_pairs=len(g),
                spearman_rho=rho,top_pair_rank=int(ranks[g.is_top_k].iloc[0]),
                top_pair_value=float(g.loc[g.is_top_k,m].iloc[0]),
                top_pair_mean_similarity=float(g.loc[g.is_top_k,'mean_similarity'].iloc[0]),
                top_pair_cell_types=g.loc[g.is_top_k,'source_cell_types'].iloc[0],
                descriptive_only=True))
    if set(pairs.reference)!=set(REFS):raise RuntimeError('Missing K2 pair scores for a reference')
    for m in METRICS:pairs[m+'_rank']=pairs[m+'_rank'].astype(int)
    return pairs,pd.DataFrame(rows)


def make_figures(args,summary,sweep,bench,ranking,weights,selection,pairs,pair_correlations,gains,dilution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter,PercentFormatter,ScalarFormatter
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.75,'axes.titleweight':'bold','legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white','axes.unicode_minus':True})
    figdir=args.outdir/'figures';figdir.mkdir(exist_ok=True)
    ks=np.arange(1,args.k_max+1)
    def save(fig,name):
        for ext in ['pdf','svg','png']:
            fig.savefig(figdir/f'{name}.{ext}',dpi=args.dpi,bbox_inches='tight',pad_inches=.12)
        plt.close(fig)
    def panel(ax,label,title):
        ax.set_title(f'{label}  {title}',loc='left',pad=9)
    def kaxis(ax):
        ax.set_xticks(ks);ax.set_xlabel('Number of source cell types (K)');ax.grid(axis='y',color='#DCE1E6',lw=.6);ax.set_axisbelow(True)
    def fmt(ax,m,gain=False):
        ax.yaxis.set_major_locator(MaxNLocator(5));formatter=ScalarFormatter(useOffset=False);formatter.set_powerlimits((-3,4));ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(('MSE reduction' if m=='mse' else 'Δ Pearson r') if gain else ('MSE (↓)' if m=='mse' else 'Pearson r (↑)'))
    def get(fam,ref,wt=None):
        x=summary[summary.family.eq(fam)&summary.reference.eq(ref)]
        if wt is not None:x=x[x.weighting.eq(wt)]
        return x.sort_values('k')
    def base(fam,ref,m):
        d=get(fam,ref)
        return float(d[m].iloc[0]) if len(d) else None
    def common_title(fig,title,sub):
        fig.suptitle(title,fontsize=15,weight='bold',y=.985)
        fig.text(.5,.937,sub,ha='center',va='top',fontsize=9,color='#475569')
    ref='real_PBS'
    # FIG 1: matched-K random selection control. Rankings + three metric panels.
    if len(selection):
        fig,axs=plt.subplots(2,2,figsize=(10.6,7.4));fig.subplots_adjust(left=.20,right=.97,bottom=.20,top=.84,hspace=.63,wspace=.42)
        common_title(fig,'Does source-cell similarity improve perturbation transfer?',f'{args.holdout} held out | frozen shared Z200 | real PBS reference | mean over {len(args.cytokines)} cytokines')
        ax=axs[0,0];rr=ranking.sort_values('rank',ascending=False)
        ax.barh(rr.cell_type,rr.similarity,color=[PROP if r<=2 else '#A6B8C8' for r in rr['rank']],height=.68)
        ax.axvline(0,color=GRAY,lw=.7);ax.set_xlabel('Cosine similarity of PBS latent centroids');ax.tick_params(axis='y',labelsize=7.5)
        lo=min(0,float(rr.similarity.min())-.05);hi=max(.1,float(rr.similarity.max())+.12);ax.set_xlim(lo,hi)
        for i,v in enumerate(rr.similarity):ax.text(v+(.009 if v>=0 else -.009),i,f'{v:.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=7)
        panel(ax,'A','Rank source cell types')
        for j,(ax,m) in enumerate(zip([axs[0,1],axs[1,0],axs[1,1]],METRICS),1):
            d=selection[selection.reference.eq(ref)&selection.metric.eq(m)].sort_values('k'); x=d.k.to_numpy()
            ax.fill_between(x,d.random_q10.to_numpy(),d.random_q90.to_numpy(),color=RAND,alpha=.24,lw=0)
            ax.plot(x,d.random_median,'o--',color=RAND,lw=1.4,ms=3)
            ax.plot(x,d.top_k,'o-',color=PROP,lw=2,ms=4)
            b=base('matched_all_latent_Z200',ref,m)
            if b is not None:ax.axhline(b,color=GRAY,ls=':',lw=1.3)
            panel(ax,'BCD'[j-1],METRIC_LABEL[m]);kaxis(ax);fmt(ax,m)
        fig.legend(handles=[Line2D([0],[0],color=PROP,marker='o',label='Top-K, equal weights'),Line2D([0],[0],color=RAND,ls='--',marker='o',label='Random subsets: median'),plt.Rectangle((0,0),1,1,fc=RAND,alpha=.24,label='Random subset 10–90% range'),Line2D([0],[0],color=GRAY,ls=':',label='All sources, matched')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
        save(fig,'Fig1_source_selection')
    # FIG 2: performance and direct weighting effect, same source cells.
    for ref,name in [('real_PBS','Fig2_K_weighting'),('reconstructed_PBS','FigS1_K_weighting_reconstructed')]:
        fig,axs=plt.subplots(2,3,figsize=(11.7,6.5),sharex='col');fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.82,hspace=.53,wspace=.33)
        common_title(fig,'Increasing K: can weighting mitigate performance loss?',f'{args.holdout} | shared Z200 | {ref.replace("_"," ")} | paired comparison at each K')
        dd=gains[gains.family.eq('shared_latent_transfer')&gains.reference.eq(ref)]
        for j,m in enumerate(METRICS):
            ax=axs[0,j]
            for wt,col in [('equal',EQUAL),('proportional',PROP)]:
                d=get('shared_latent_transfer',ref,wt);ax.plot(d.k,d[m],'o-',color=col,lw=2,ms=4)
            b=base('matched_all_latent_Z200',ref,m)
            if b is not None: ax.axhline(b,color=GRAY,ls=':',lw=1.1)
            if args.k_max>=2:ax.axvline(2,color='#C5CCD3',lw=.9,zorder=0)
            panel(ax,'ABC'[j],METRIC_LABEL[m]);fmt(ax,m);ax.grid(axis='y',color='#E0E5E9',lw=.6);ax.set_xticks(ks)
            ax=axs[1,j];gc=m+'_gain'
            for _,g in dd.groupby('cytokine'):
                g=g.sort_values('k');ax.plot(g.k,g[gc],color='#AAB4BF',lw=.9,alpha=.7)
            means=dd.groupby('k')[gc].mean();ax.plot(means.index,means,'o-',color=PROP,lw=2.25,ms=4)
            ax.axhline(0,color=GRAY,lw=.9);panel(ax,'DEF'[j],'Proportional minus equal' if m!='mse' else 'Equal minus proportional MSE');kaxis(ax);fmt(ax,m,True)
        fig.legend(handles=[Line2D([0],[0],color=EQUAL,marker='o',label='Equal'),Line2D([0],[0],color=PROP,marker='o',label='Proportional / mean gain'),Line2D([0],[0],color=GRAY,ls=':',label='All sources, matched'),Line2D([0],[0],color='#AAB4BF',label='Individual cytokines')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False)
        save(fig,name)
    # FIG 3: only directional alignment with fixed K2 equal reference.
    fig,ax=plt.subplots(figsize=(6.5,4.65));fig.subplots_adjust(left=.14,right=.96,top=.76,bottom=.23)
    anchor_names=', '.join(ranking.sort_values('rank').cell_type.head(2))
    common_title(fig,'Alignment with the K2 latent shift',f'{args.holdout} | K2 reference: {anchor_names}\nDescriptive alignment; K2 is not ground truth')
    for wt,col in [('equal',EQUAL),('proportional',PROP)]:
        a=dilution[dilution.weighting.eq(wt)].groupby('k')['latent_cosine_to_K2'].agg(['mean','min','max'])
        ax.plot(a.index,a['mean'],'o-',color=col,lw=2,ms=4)
        ax.fill_between(a.index.to_numpy(),a['min'].to_numpy(),a['max'].to_numpy(),color=col,alpha=.12,lw=0)
    ax.axhline(1,color=GRAY,lw=.8,ls=':');ax.set_ylabel('Cosine similarity to K2 shift');kaxis(ax)
    fig.legend(handles=[Line2D([0],[0],color=EQUAL,marker='o',label='Equal'),Line2D([0],[0],color=PROP,marker='o',label='Proportional'),plt.Rectangle((0,0),1,1,fc=RAND,alpha=.18,label='Min–max across cytokines')],loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False)
    save(fig,'Fig3_signal_dilution_diagnostics')
    # FIG 4: main model comparison on common real-PBS reference, fixed K2 candidate.
    labels=[];blocks=[];colors=[]
    def add(d,label,col):
        if len(d):labels.append(label);blocks.append(d);colors.append(col)
    for fam in BENCH:
        col=RAND if fam=='original_naive' else GOLD if fam=='strong_naive_K1' else GRAY
        add(bench[bench.family.eq(fam)&bench.reference.eq('real_PBS')],BENCH[fam],col)
    for wt,col in [('equal',EQUAL),('proportional',PROP)]:
        k=min(2,args.k_max);add(sweep[sweep.family.eq('shared_latent_transfer')&sweep.reference.eq('real_PBS')&sweep.k.eq(k)&sweep.weighting.eq(wt)],f'Shared Z200 | K{k}, {wt}',col)
    fig,axs=plt.subplots(1,3,figsize=(11.8,5.7),sharey=True);fig.subplots_adjust(left=.30,right=.98,bottom=.15,top=.80,wspace=.36)
    common_title(fig,'Comparison with original and learning-free baselines',f'{args.holdout} | real PBS | points: individual cytokines; diamonds: unweighted mean')
    for j,(ax,m) in enumerate(zip(axs,METRICS)):
        for i,(d,col) in enumerate(zip(blocks,colors)):
            v=d[m].to_numpy();y=i+np.linspace(-.16,.16,len(v));ax.scatter(v,y,s=13,color=col,alpha=.45,zorder=2);ax.scatter(np.nanmean(v),i,s=45,marker='D',color=col,edgecolor='white',lw=.5,zorder=3)
        ax.set_yticks(range(len(labels)));ax.tick_params(axis='y',labelleft=j==0,length=0);ax.grid(axis='x',color='#DCE1E6',lw=.6);ax.set_axisbelow(True);ax.set_xlabel('MSE (↓)' if m=='mse' else 'Pearson r (↑)');ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f' if m=='mse' else '%.2f'));panel(ax,'ABC'[j],METRIC_LABEL[m])
    # Shared axis label visibility is set separately; set common labels once.
    axs[0].set_yticks(range(len(labels)),labels);axs[0].invert_yaxis()
    save(fig,'Fig4_model_comparison_real_PBS')
    # FIG 5: actual exhaustive pair observations, not simulated points.
    d=pairs[pairs.reference.eq('real_PBS')].copy()
    correlations=pair_correlations[pair_correlations.reference.eq('real_PBS')].set_index('metric')
    n=len(d);top=d[d.is_top_k];other=d[~d.is_top_k]
    fig,axs=plt.subplots(1,3,figsize=(11.2,4.55));fig.subplots_adjust(left=.085,right=.98,bottom=.24,top=.75,wspace=.36)
    common_title(fig,'Source-pair similarity and prediction accuracy',f'{args.holdout} | all {n} source pairs | equal weights | real PBS | means over {len(args.cytokines)} cytokines')
    for j,(ax,m) in enumerate(zip(axs,METRICS)):
        ax.scatter(other.mean_similarity,other[m],s=23,color='#8193A6',alpha=.70,edgecolors='white',linewidths=.35,zorder=2)
        ax.scatter(top.mean_similarity,top[m],s=88,marker='D',color=PROP,edgecolors='white',linewidths=.9,zorder=4)
        rho=float(correlations.loc[m,'spearman_rho']);rank=int(correlations.loc[m,'top_pair_rank'])
        text=f'Spearman ρ = {rho:.3f}\nTop pair rank: {rank}/{n}' if np.isfinite(rho) else f'Spearman ρ undefined\nTop pair rank: {rank}/{n}'
        # Annotation above the axes so no data points are obscured.
        ax.text(0,1.025,text,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.2,color=GRAY)
        ax.set_title(f'{"ABC"[j]}  {METRIC_LABEL[m]}',loc='left',pad=37)
        ax.set_xlabel('Mean source-to-target\nlatent cosine similarity');fmt(ax,m)
        ax.grid(color='#E0E5E9',lw=.6);ax.set_axisbelow(True);ax.margins(x=.08,y=.12)
    topname=top.source_cell_types.iloc[0].replace(' | ',' + ')
    fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='None',color='#8193A6',label=f'Other {n-1} pairs'),Line2D([0],[0],marker='D',linestyle='None',color=PROP,label=f'Two closest sources: {topname}')],loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    save(fig,'Fig5_source_pair_similarity_accuracy')
    # FIG S2: exact weights, readable and editable.
    order=ranking.sort_values('rank').cell_type.tolist()
    fig,axs=plt.subplots(1,2,figsize=(10.9,5.7),sharey=True);fig.subplots_adjust(left=.20,right=.89,bottom=.13,top=.79,wspace=.18)
    common_title(fig,'Source-cell weights across the K sweep',f'{args.holdout} | latent PBS cosine ranking | percentages of total transferred shift')
    for j,(ax,wt) in enumerate(zip(axs,['equal','proportional'])):
        d=weights[weights.weighting.eq(wt)].pivot(index='cell_type',columns='k',values='weight').reindex(index=order,columns=ks)
        im=ax.imshow(d.to_numpy(),aspect='auto',cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(len(ks)),ks);ax.set_yticks(range(len(order)),order);ax.tick_params(axis='y',labelleft=j==0);ax.set_xlabel('K');panel(ax,'AB'[j],wt.capitalize()+' weights')
        for y in range(len(order)):
            for x in range(len(ks)):
                v=d.iloc[y,x]
                if v>0:ax.text(x,y,f'{100*v:.0f}',ha='center',va='center',fontsize=7,color='white' if v>.48 else '#17304A')
    cax=fig.add_axes([.92,.22,.018,.51]);fig.colorbar(im,cax=cax,label='Weight fraction');save(fig,'FigS2_source_weights')
    # FIG S3: individual cytokine response to weighting; diverging about zero.
    fig,axs=plt.subplots(1,3,figsize=(11.4,4.15));fig.subplots_adjust(left=.12,right=.97,bottom=.21,top=.72,wspace=.53)
    common_title(fig,'Cytokine-level effect of proportional weighting',f'{args.holdout} | real PBS | positive values favor proportional weighting')
    d=gains[gains.family.eq('shared_latent_transfer')&gains.reference.eq('real_PBS')]
    for j,(ax,m) in enumerate(zip(axs,METRICS)):
        a=d.pivot(index='cytokine',columns='k',values=m+'_gain').reindex(index=args.cytokines,columns=ks);vv=a.to_numpy();bound=max(np.nanmax(np.abs(vv)),1e-8)
        im=ax.imshow(vv,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
        ax.set_xticks(range(len(ks)),ks);ax.set_yticks(range(len(a)),a.index if j==0 else ['']*len(a));ax.set_xlabel('K');panel(ax,'ABC'[j],METRIC_LABEL[m]);fig.colorbar(im,ax=ax,shrink=.83,pad=.04,format='%.3f' if m=='mse' else '%.2f')
    save(fig,'FigS3_cytokine_weighting_gain')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=Path('data/hcd_subset.h5ad'))
    p.add_argument('--shared-z100',type=Path,default=Path('data/shared_CD14_Mono'))
    p.add_argument('--shared-z200',type=Path,default=Path('data/shared_CD14_Mono_Z200'))
    p.add_argument('--holdout',default='CD14 Mono')
    p.add_argument('--cytokines',nargs='+',default=CYTOKINES)
    p.add_argument('--k-max',type=int,default=10)
    p.add_argument('--random-subsets',type=int,default=24,help='Uniform unique subsets per K>2; K1/K2 always exhaustive. 0 skips K>2 subsets.')
    p.add_argument('--seed',type=int,default=42,help='Random-subset sampling seed; no training.')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--decode-batch-size',type=int,default=512)
    p.add_argument('--keep-rare',action='store_true',help='Use only if the saved shared checkpoint was trained with --keep-rare')
    p.add_argument('--historical-csv',type=Path,default=Path('analysis/results_L2_vs_L4_H800_Z100.csv'))
    p.add_argument('--historical-run',default='scgen_runs_L2_H800_Z100')
    p.add_argument('--previous-ablation-csv',type=Path,default=None,help='Optional prior comprehensive results.csv: verify original naive and strong gene-cosine K1 against the previous run; never import or duplicate the benchmark.')
    p.add_argument('--outdir',type=Path,default=Path('data/similarity_transfer_Z200'))
    p.add_argument('--dpi',type=int,default=600,help='Raster export DPI; PDF/SVG are vector.')
    p.add_argument('--no-figures',action='store_true')
    args=p.parse_args()
    if args.k_max<2 or args.random_subsets<0 or args.decode_batch_size<1 or args.dpi<72:p.error('Require K>=2, random-subsets>=0, positive batch size and DPI>=72')
    if args.seed<0:p.error('Nonnegative seed required')
    if 'PBS' in args.cytokines or len(set(args.cytokines))!=len(args.cytokines):p.error('Specify unique treatment cytokines, excluding PBS')
    if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())):p.error('Choose a new or empty --outdir; existing outputs are not overwritten.')
    for r in [args.shared_z100,args.shared_z200]:
        if args.outdir.resolve()==r.resolve() or r.resolve() in args.outdir.resolve().parents:p.error('Output must be separate from saved model runs')
    tag='shared__'+args.holdout.replace(' ','_').replace('/','-')
    checkpoint=args.shared_z200/tag
    if not (checkpoint/'model.pt').is_file():p.error(f'Missing Z200 checkpoint: {checkpoint}')
    tables={}
    for label,path,run in [('original_scgen_Z100',args.historical_csv,args.historical_run),('shared_scgen_Z100',args.shared_z100/'results.csv',None),('shared_scgen_Z200',args.shared_z200/'results.csv',None)]:
        tables[label]=read_baselines(path,args.holdout,args.cytokines,run)
    prior=None
    if args.previous_ablation_csv is not None:
        prior=pd.read_csv(args.previous_ablation_csv)
        prior=prior[prior.holdout.eq(args.holdout)&prior.cytokine.isin(args.cytokines)&prior.reference.eq('real_PBS')]
        pn=prior[prior.family.eq('original_naive')]
        pk=prior[prior.family.eq('gene_similarity_gene_transfer')&prior.k.eq(1)&prior.weighting.eq('equal')&prior.similarity_metric.eq('cosine')&prior.similarity_space.eq('gene')]
        for d in [pn,pk]:
            if len(d)!=len(args.cytokines) or d.cytokine.duplicated().any() or set(d.cytokine)!=set(args.cytokines):raise ValueError('Prior ablation CSV must contain unique original naive and VAE-free cosine K1 scores for every cytokine')
            if not np.isfinite(d[METRICS].to_numpy()).all():raise ValueError('Invalid prior scores')
        prior=(pn.set_index('cytokine'),pk.set_index('cytokine'))
    import anndata as ad
    import torch
    import scgen
    from scgen._scgenvae import SCGENVAE
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA requested but not available')
    # Same compatibility patch as original shared training and previous successful ablation.
    original_inference=SCGENVAE.inference
    def inference_fixed(self,x):
        o=original_inference(self,x)
        if 'qz_m' in o:o.setdefault('qzm',o['qz_m'])
        if 'qz_v' in o:o.setdefault('qzv',o['qz_v'])
        return o
    SCGENVAE.inference=inference_fixed
    args.outdir.mkdir(parents=True,exist_ok=True)
    print('Loading original HCD input; no renormalization, new HVGs or donor splitting.')
    data=ad.read_h5ad(args.data)
    data=data[data.obs['cytokine'].isin(['PBS']+args.cytokines)].copy()
    if not args.keep_rare:data=data[~data.obs['cell_type'].isin(RARE)].copy()
    data.obs['condition']=data.obs['cytokine'].astype(str);data.obs['cell_type']=data.obs['cell_type'].astype(str)
    if 'counts' in data.layers:del data.layers['counts']
    ct=data.obs['cell_type'].to_numpy();cond=data.obs['condition'].to_numpy();sources=sorted(set(ct)-{args.holdout})
    if args.holdout not in set(ct) or args.k_max>len(sources):raise ValueError(f'Missing holdout or K exceeds {len(sources)} available sources')
    for t in [args.holdout]+sources:
        for c in ['PBS']+args.cytokines:
            if not ((ct==t)&(cond==c)).any():raise ValueError(f'Missing group {t}/{c}; refusing to alter original split')
    if data.n_vars!=7000:warnings.warn(f'Expected 7000 genes; input has {data.n_vars}.')
    def mean_x(t,c):return dense(data.X[(ct==t)&(cond==c)]).mean(0)
    ctrl=mean_x(args.holdout,'PBS');gene_pbs=np.stack([mean_x(t,'PBS') for t in sources])
    gene_shifts={c:np.stack([mean_x(t,c)-gene_pbs[i] for i,t in enumerate(sources)]) for c in args.cytokines}
    # Strong learning-free K1: select on PBS gene means ONLY, no model or
    # latent features. Fix this source before using held-out treated data.
    gene_cosine=cosine(gene_pbs,ctrl)
    gene_order=np.argsort(-gene_cosine,kind='stable')
    gene_rank=np.empty(len(sources),int);gene_rank[gene_order]=np.arange(1,len(sources)+1)
    gene_k1=int(gene_order[0])
    gene_ranking=pd.DataFrame(dict(holdout=args.holdout,cell_type=sources,rank=gene_rank,
        similarity=gene_cosine,selected_K1=gene_rank==1)).sort_values('rank')
    save_csv(gene_ranking,args.outdir/'strong_naive_K1_gene_similarity.csv')
    print(f'Strong naive K1 (gene cosine; no VAE): {sources[gene_k1]}, similarity={gene_cosine[gene_k1]:.6f}')
    # No target treated cells enter the encoder, similarity, source shifts or decoder inputs.
    train=data[~((ct==args.holdout)&(cond!='PBS'))].copy()
    train.X=dense(train.X).astype(np.float32,copy=False)
    tc=train.obs['cell_type'].to_numpy();cc=train.obs['condition'].to_numpy()
    assert not ((tc==args.holdout)&(cc!='PBS')).any()
    print(f'Loading frozen Z200 model: {checkpoint}; {train.n_obs:,} training cells, {len(sources)} source types.')
    model=scgen.SCGEN.load(str(checkpoint),adata=train,accelerator='cpu',device='auto')
    model.to_device(args.device);model.module.eval()
    for par in model.module.parameters():par.requires_grad_(False)
    with torch.inference_mode():z=np.asarray(model.get_latent_representation(),dtype=np.float32)
    if z.shape!=(train.n_obs,200) or not np.isfinite(z).all():raise RuntimeError(f'Expected valid Z200 embeddings, received {z.shape}')
    target_z=z[(tc==args.holdout)&(cc=='PBS')];pbs_z=np.stack([z[(tc==t)&(cc=='PBS')].mean(0) for t in sources])
    latent_shifts={c:np.stack([z[(tc==t)&(cc==c)].mean(0) for t in sources])-pbs_z for c in args.cytokines}
    recon=decode_mean(model,target_z,torch,args.decode_batch_size)
    if not np.isfinite(recon).all():raise RuntimeError('Nonfinite PBS reconstruction')
    s=cosine(pbs_z,target_z.mean(0));order,rank,configs,random_configs,random_meta=make_design(s,sources,args.k_max,args.random_subsets,args.seed)
    ranking=pd.DataFrame(dict(holdout=args.holdout,cell_type=sources,rank=rank,similarity=s)).sort_values('rank')
    weight_rows=[]
    for d in configs:
        w=d['w']
        for i,t in enumerate(sources):weight_rows.append(dict(holdout=args.holdout,k=d['k'],weighting=d['weighting'],cell_type=t,rank=rank[i],similarity=s[i],selected=bool(i in d['indices']),weight=w[i],fallback_equal=d['fallback']))
    weights=pd.DataFrame(weight_rows)
    save_csv(ranking,args.outdir/'similarities.csv');save_csv(weights,args.outdir/'weights.csv')
    if len(random_meta):save_csv(random_meta,args.outdir/'random_subsets.csv')
    np.savez_compressed(args.outdir/'source_shifts.npz',cell_types=np.array(sources),cytokines=np.array(args.cytokines),genes=np.array(data.var_names.astype(str),dtype=str),pbs_latent_centroids=pbs_z,target_pbs_latent_centroid=target_z.mean(0),cosine=s,latent_shifts=np.stack(list(latent_shifts.values())),gene_shifts=np.stack(list(gene_shifts.values())))
    print(ranking[['rank','cell_type','similarity']].to_string(index=False))
    print(f'{len(configs)} top-K settings and {len(random_configs)} distinct matched-size random control subsets. No training.')
    rows=[];benchmark=[];random_rows=[];dilution=[];checks=[]
    # Target truth is accessed only after all similarities and weights are fixed.
    for c in args.cytokines:
        gp=gene_shifts[c];lp=latent_shifts[c];anchor=lp[order[:2]].mean(0)
        real_delta=mean_x(args.holdout,c)-ctrl
        naive=score(ctrl+gp.mean(0),ctrl,real_delta)
        check=dict(cytokine=c,holdout=args.holdout,n_target_PBS=len(target_z),n_target_treated=int(((ct==args.holdout)&(cond==c)).sum()))
        benchmark.append(result(c,args.holdout,'original_naive','real_PBS',naive,weighting='all_equal'))
        strong=score(ctrl+gp[gene_k1],ctrl,real_delta)
        benchmark.append(result(c,args.holdout,'strong_naive_K1','real_PBS',strong,k=1,weighting='equal'))
        for label,table in tables.items():
            ok=matches(naive,table.loc[(c,'naive')]);check[label+'_naive_matches']=ok
            if not ok:raise RuntimeError(f'Original naive mismatch: {label}/{c}; verify input')
            for method,ref in [('scGen_vs_real_ctrl','real_PBS'),('scGen','reconstructed_PBS')]:
                benchmark.append(result(c,args.holdout,label,ref,table.loc[(c,method),METRICS].astype(float).to_dict(),origin='historical' if label=='original_scgen_Z100' else 'saved'))
        if prior is not None:
            ok=matches(naive,prior[0].loc[c]);check['prior_ablation_naive_matches']=ok
            if not ok:raise RuntimeError(f'Prior ablation original naive mismatch: {c}')
            ok=matches(strong,prior[1].loc[c]);check['prior_ablation_strong_naive_K1_matches']=ok
            if not ok:raise RuntimeError(f'Prior ablation strong naive K1 mismatch: {c}')
        filetag=c+'__'+args.holdout.replace(' ','_').replace('/','-');rf=args.shared_z200/(filetag+'_recon.h5ad');pf=args.shared_z200/(filetag+'_pred.h5ad')
        check['reconstruction_checked']=rf.is_file();check['saved_prediction_checked']=rf.is_file() and pf.is_file()
        if rf.is_file():
            rr=ad.read_h5ad(rf)
            if not rr.var_names.equals(data.var_names) or rr.n_obs!=len(target_z):raise RuntimeError(f'Reconstruction dimensions/gene order mismatch: {rf}')
            sr=dense(rr.X).mean(0);del rr
            check['reconstruction_mean_mse']=float(np.mean((sr-recon)**2))
            if check['reconstruction_mean_mse']>1e-7:raise RuntimeError(f'Saved PBS reconstruction differs: {c}')
            if pf.is_file():
                pp=ad.read_h5ad(pf)
                if not pp.var_names.equals(data.var_names) or pp.n_obs!=len(target_z):raise RuntimeError(f'Prediction dimensions/gene order mismatch: {pf}')
                pm=dense(pp.X).mean(0);del pp
                for method,base in [('scGen',sr),('scGen_vs_real_ctrl',ctrl)]:
                    ok=matches(score(pm,base,real_delta),tables['shared_scgen_Z200'].loc[(c,method)]);check['saved_Z200_'+method+'_matches']=ok
                    if not ok:raise RuntimeError(f'Saved Z200 prediction score mismatch: {c}/{method}')
        checks.append(check)
        cache={}
        def predict(w):
            delta=(w@lp).astype(np.float32);key=delta.tobytes()
            if key not in cache:
                pm=decode_mean(model,target_z+delta[None,:],torch,args.decode_batch_size)
                if not np.isfinite(pm).all():raise RuntimeError('Nonfinite decoder output')
                cache[key]=pm
            return cache[key]
        allp=predict(np.ones(len(sources))/len(sources))
        for ref,base in [('real_PBS',ctrl),('reconstructed_PBS',recon)]:benchmark.append(result(c,args.holdout,'matched_all_latent_Z200',ref,score(allp,base,real_delta),weighting='all_equal'))
        for d in configs:
            k=d['k'];wt=d['weighting'];w=d['w'];pred=predict(w)
            for ref,base in [('real_PBS',ctrl),('reconstructed_PBS',recon)]:rows.append(result(c,args.holdout,'shared_latent_transfer',ref,score(pred,base,real_delta),k,wt))
            dilution.append(dilution_row(c,args.holdout,k,wt,w,s,lp,anchor,order,pred,ctrl,recon,real_delta))
        for i,d in enumerate(random_configs,1):
            pred=predict(d['w'])
            for ref,base in [('real_PBS',ctrl),('reconstructed_PBS',recon)]:
                r=result(c,args.holdout,'random_latent_transfer',ref,score(pred,base,real_delta),d['k'],'equal');r['subset_id']=d['subset_id'];random_rows.append(r)
            if i%100==0:print(f'  {c}: {i}/{len(random_configs)} matched random subsets')
        print(f'{c}: completed, {len(cache)} unique latent shifts decoded; naive checks passed.')
        save_csv(pd.DataFrame(rows),args.outdir/'sweep_results.csv');save_csv(pd.DataFrame(benchmark),args.outdir/'benchmark_results.csv');save_csv(pd.DataFrame(checks),args.outdir/'validation.csv')
        if random_rows:save_csv(pd.DataFrame(random_rows),args.outdir/'random_subset_results.csv')
    sweep=pd.DataFrame(rows);bench=pd.DataFrame(benchmark);results=pd.concat([bench,sweep],ignore_index=True);summary=summarize(results)
    if not summary.n_cytokines.eq(len(args.cytokines)).all():raise RuntimeError('Incomplete cytokine groups')
    for m in METRICS:
        if not summary[m+'_n'].eq(len(args.cytokines)).all():warnings.warn(f'Some {m} values undefined; valid counts reported, do not interpret incomplete means as matched.')
    save_csv(results,args.outdir/'results.csv');save_csv(summary,args.outdir/'summary.csv');save_csv(summarize(bench),args.outdir/'benchmark_summary.csv')
    # Same neighbors, same VAE, only weighting differs. Positive gain always better.
    keys=['holdout','cytokine','family','reference','k']
    gains=paired(sweep[sweep.weighting.eq('proportional')],sweep[sweep.weighting.eq('equal')],keys,'proportional_minus_equal')
    gs=gain_summary(gains,['holdout','family','reference','k'])
    save_csv(gains,args.outdir/'weighting_effect.csv');save_csv(gs,args.outdir/'weighting_effect_summary.csv')
    # Explicit K2 anchor comparison: how much performance changes as K grows.
    a=sweep[sweep.family.eq('shared_latent_transfer')]
    anchor=a[a.k.eq(2)&a.weighting.eq('equal')][['holdout','cytokine','reference']+METRICS]
    d=a.merge(anchor,on=['holdout','cytokine','reference'],suffixes=('_test','_control'),validate='many_to_one')
    for m in METRICS:d[m+'_gain']=(-1 if m=='mse' else 1)*(d[m+'_test']-d[m+'_control'])
    save_csv(d,args.outdir/'performance_relative_to_K2.csv');save_csv(gain_summary(d,['holdout','reference','k','weighting']),args.outdir/'performance_relative_to_K2_summary.csv')
    dil=pd.DataFrame(dilution);save_csv(dil,args.outdir/'signal_dilution.csv')
    dcols=[x for x in dil.columns if x not in ['holdout','cytokine','k','weighting']]
    dsum=dil.groupby(['holdout','k','weighting'],sort=False)[dcols].agg(['mean','min','max','std']);dsum.columns=['_'.join(x) for x in dsum.columns];save_csv(dsum.reset_index(),args.outdir/'signal_dilution_summary.csv')
    random_means,selection=selection_summaries(pd.DataFrame(random_rows),random_meta,sweep)
    if len(selection):save_csv(random_means,args.outdir/'random_subset_means.csv');save_csv(selection,args.outdir/'source_selection_summary.csv')
    pairs,pair_correlations=analyze_pairs(random_means,len(sources),len(args.cytokines))
    # The enumerated top pair must reproduce the direct equal-K2 sweep.
    for ref in REFS:
        top=pairs[pairs.reference.eq(ref)&pairs.is_top_k].iloc[0]
        expected=summary[summary.family.eq('shared_latent_transfer')&summary.reference.eq(ref)&summary.k.eq(2)&summary.weighting.eq('equal')].iloc[0]
        if not np.allclose(top[METRICS].to_numpy(dtype=float),expected[METRICS].to_numpy(dtype=float),rtol=0,atol=1e-9):
            raise RuntimeError('Enumerated top pair does not reproduce the direct K2 equal result')
    save_csv(pairs,args.outdir/'source_pair_similarity_accuracy.csv')
    save_csv(pair_correlations,args.outdir/'source_pair_spearman.csv')
    # Explore the best K by metric, but do not silently choose it for future holdouts.
    best=[]
    for _,g in summary[summary.family.eq('shared_latent_transfer')].groupby(['holdout','family','reference','weighting']):
        for m in METRICS:
            a=g[g[m+'_n'].eq(len(args.cytokines))]
            if len(a):r=a.sort_values([m,'k'],ascending=[m=='mse',True]).iloc[0].to_dict();r['selected_metric']=m;r['exploratory_only']=True;best.append(r)
    save_csv(pd.DataFrame(best),args.outdir/'best_K_exploratory.csv')
    # Compare top-K against every existing benchmark, reference matched only.
    bc=[]
    for fam,g in bench.groupby('family'):
        for ref,h in g.groupby('reference'):
            sub=sweep[sweep.reference.eq(ref)]
            x=sub.merge(h[['holdout','cytokine','reference']+METRICS],on=['holdout','cytokine','reference'],suffixes=('_test','_control'),validate='many_to_one');x['benchmark']=fam
            for m in METRICS:x[m+'_gain']=(-1 if m=='mse' else 1)*(x[m+'_test']-x[m+'_control'])
            bc.append(x)
    bc=pd.concat(bc,ignore_index=True);save_csv(bc,args.outdir/'benchmark_comparisons.csv');save_csv(gain_summary(bc,['holdout','family','reference','k','weighting','benchmark']),args.outdir/'benchmark_comparison_summary.csv')
    def ver(pkg):
        try:return version(pkg)
        except PackageNotFoundError:return None
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};cfg.update(n_genes=data.n_vars,n_train_cells=train.n_obs,n_source_types=len(sources),source_types=sources,latent_dim=200,similarity='cosine_of_PBS_latent_centroids',strong_naive_similarity='cosine_of_PBS_gene_centroids',strong_naive_source=sources[gene_k1],strong_naive_cosine=float(gene_cosine[gene_k1]),n_random_subsets=len(random_configs),training_performed=False,target_treated_used_for_weights=False,versions={x:ver(x) for x in ['numpy','pandas','anndata','torch','scgen','scvi-tools','matplotlib']})
    (args.outdir/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (args.outdir/'METHODS.txt').write_text('''Focused inference-only ablation of source selection and weighting.
Original pooled-donor cell-type holdout, rare-cell filter and log-normalized HCD X are unchanged. Frozen Z200 uses PBS from all types and treated cells from non-target types. Target treated cells are used only for scoring; Z100 and original cytokine-specific Z100 scores are read, not retrained. No Z400. All latent-transfer similarities are cosine of PBS posterior-mean latent centroids. The original naive is VAE-free, equal average of gene shifts from all eligible sources. Strong naive K1 independently ranks source populations by cosine similarity of PBS mean log-normalized gene expression (the same HVGs), selects the single closest source, and adds its mean treated-minus-PBS gene shift to target mean PBS expression. This baseline uses no VAE outputs. No gene-space K sweep. Optional previous ablation scores are used only to verify these two naive baselines, not imported as additional results.
Top K: descending cosine similarity, ties broken alphabetically. Equal weights: 1/K. Proportional: positive cosine divided by sum of positive cosines within selected top K; nonpositive values receive zero, equal fallback if none positive. K1 must coincide. For each cytokine, each source shift is its treated centroid minus its own PBS centroid. Combine shifts by the selected weights. Add the resulting latent vector to each target PBS cell; decode individually then average. No model updates. All-source matched shift is a separate baseline, distinct from default scGen.predict().
Selection control: all possible subsets at K1 and K2; at larger K, up to --random-subsets distinct subsets sampled uniformly without replacement using --seed. Same subsets across cytokines; equal weighting in both top-K and random groups. Includes top-K if sampled/enumerated. Random distribution is of subset means over cytokines, not confidence intervals or biological replicates. The rank fractions are descriptive, not inferential p-values. Random equal weights isolate source selection at fixed K. For the exhaustive K2 analysis, pair similarity is the arithmetic mean of the two source-to-target PBS latent cosines, not the similarity between the two sources. Spearman rank correlation is calculated between this quantity and each metric averaged over cytokines. Pairs overlap and are not independent observations; report descriptive rho and ranks without inferential p-values.
Evaluation: real delta is mean real treated minus mean real PBS. Predicted delta subtracts real PBS (primary common reference) or reconstructed PBS (original scGen convention, supplementary). Pearson all, Pearson top100 by absolute real delta, and original MSE. No clipping. Mean metrics are unweighted across cytokines, not independent donor replicates; report valid counts. Paired effects are positive for better correlations or lower MSE. Historical benchmarks validated via original naive at CSV rounding precision; this does not prove identical input cell IDs. Saved predictions/reconstructions checked when present.
Signal dilution diagnostics: cumulative weight on fixed top2 source set; weighted similarity; effective source number 1/sum(w^2); latent shift norm, alignment and norm relative to K2 equal shift; cancellation index ||sum w*delta||/sum w*||delta|| (0 to 1); decoded response norm / real response norm. K2 is an exploratory anchor, NOT the ground-truth response. Changes can be consistent with dilution but do not by themselves establish a causal mechanism; proportional weighting may help at some K/metrics and hurt others. Report actual signed gains, not presumed improvements. Best K chosen on these CD14 Mono results is exploratory and must be validated on other held-out cell types/seeds. No donor splitting added.
''')
    (args.outdir/'FIGURE_GUIDE.txt').write_text('''Figure 1 | Source selection at matched K. PBS latent cosine ranking, top-K equal weights vs median and 10–90% range of random subset mean scores. Exact enumeration for K1/K2, sampled larger K. All-source matched latent benchmark. The random bands are descriptive, not confidence intervals.
Figure 2 | Equal vs proportional weighting across K, with within-cytokine paired gains below (positive always favors proportional). Real PBS reference. The light line marks K2 as candidate; no claim that it is universally optimal.
Figure 3 | Only cosine alignment with the fixed equal-weight K2 shift. K2 is not ground truth. Shading is cytokine min–max, not a confidence interval. Other descriptive diagnostics are retained in CSV files, not plotted.
Figure 4 | Eight model configurations under common real-PBS scoring: original naive, strong learning-free K1, original scGen Z100, shared Z100/Z200, all-source matched latent transfer, and K2 equal/proportional. Dots are individual cytokine scores; diamonds are unweighted means. Both naive methods are learning-free.
Figure 5 | All possible equal-weight K2 source pairs: mean source-to-target PBS latent cosine versus mean cytokine prediction metrics; the two closest sources are highlighted. Descriptive Spearman rho and top-pair rank are recalculated. Overlapping pairs are not independent samples.
Supplement 1 | Same weighting analysis using reconstructed PBS (scGen original convention).
Supplement 2 | Actual source weights (%) for every K, equal and proportional.
Supplement 3 | Cytokine-specific proportional-weighting gains. Positive is better including MSE reduction.
PDF and SVG: vector exports; PNG: high-resolution raster. Colors denote methods, not significance.
''')
    if not args.no_figures:make_figures(args,summary,sweep,bench,ranking,weights,selection,pairs,pair_correlations,gains,dil)
    print('\nExhaustive K2 pair association (real PBS; descriptive):')
    print(pair_correlations.query('reference == "real_PBS"')[['metric','n_pairs','spearman_rho','top_pair_rank']].to_string(index=False))
    print('\nPrimary benchmark means (real PBS):')
    print(summarize(bench).query('reference == "real_PBS"')[['family']+METRICS].to_string(index=False))
    print('\nTop-K shared transfer, real PBS:')
    print(summary[summary.family.eq('shared_latent_transfer')&summary.reference.eq('real_PBS')][['k','weighting']+METRICS].to_string(index=False))
    print(f'\nSaved {len(results)} top-K/benchmark evaluation rows and {len(random_rows)} random-subset rows to {args.outdir.resolve()}. No VAE training.')


if __name__=='__main__':
    main()
