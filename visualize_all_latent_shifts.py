#!/usr/bin/env python3
"""Z200 UMAP and latent-perturbation visualization: shared scGen versus learned hybrid.

Loads trained checkpoints directly; no training, decoding, re-encoding predictions,
new HVGs, normalization, donor split, or similarity-guided source retrieval.
Uses saved benchmark scGen shifts and the exact trained hybrid residual head.
One reference-fitted Z200 UMAP for all models, plus a separate target-only UMAP
for retrospective visualization, with target-only overview and per-cytokine plots.
Fixed-order and cosine-clustered heatmaps are both saved.
All predictions are fixed before observing target
treated cells. Figures are PNG only. Run from the repository root; see --help.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

CYTOKINES = ['IFN-beta', 'IFN-gamma', 'IL-1-beta', 'IL-15', 'IL-32-beta', 'IL-4']
RARE = ['pDC', 'HSPC', 'ILC', 'Plasmablast']
HEADS = ['hybrid']
COND_COLORS = {'PBS': '#8B929B', 'IFN-beta': '#D55E00', 'IFN-gamma': '#0072B2',
               'IL-1-beta': '#CC79A7', 'IL-15': '#009E73', 'IL-32-beta': '#E69F00', 'IL-4': '#7355A6'}
MODEL_COLORS = {'default': '#34699A', 'hybrid': '#8055A1', 'observed': '#222A33'}
MODEL_NAMES = {'hybrid': 'Learned hybrid'}
PBS_COLOR = '#89939F'
OBS_COLOR = '#222A33'

def dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)

def slug(s):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', s)

class PerturbationHead(nn.Module):
    """Exact hybrid architecture used by the fixed-epoch training script."""
    def __init__(self, latent_dim, n_cytokines, embedding_dim=32, hidden=256,
                 dropout=0.1, kind='hybrid'):
        super().__init__()
        if kind != 'hybrid':
            raise ValueError('This visualization supports only the hybrid head.')
        self.kind = kind
        self.embedding = nn.Embedding(n_cytokines, embedding_dim)
        self.net = nn.Sequential(nn.Linear(latent_dim + embedding_dim, hidden),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, latent_dim))
        self.global_net = nn.Linear(embedding_dim, latent_dim)

    def shift_components(self, z, cytokine_idx):
        """Global branch and nonlinear correction share the same embedding."""
        ids = torch.full((len(z),), int(cytokine_idx), dtype=torch.long, device=z.device)
        e = self.embedding(ids)
        return self.global_net(e), self.net(torch.cat([z, e], dim=1))

    def shift(self, z, cytokine_idx):
        global_shift, correction = self.shift_components(z, cytokine_idx)
        return global_shift + correction

    def forward(self, z, cytokine_idx):
        return z + self.shift(z, cytokine_idx)


def model_digest(module):
    """Check all VAE parameters AND buffers, including batch-normalization state."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()



def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)


def stratified_ids(obs, mask, cap, seed):
    rng=np.random.default_rng(seed)
    ids=[]
    for _,g in obs.loc[mask].groupby(['cell_type','cytokine','status'],sort=True):
        idx=g.index.to_numpy()
        if cap and len(idx)>cap: idx=rng.choice(idx,cap,replace=False)
        ids.extend(idx.tolist())
    return np.array(sorted(ids),dtype=int)


def umap_fit(z, neighbors, min_dist, metric, seed):
    try:
        import umap
    except ImportError as e:
        raise SystemExit('Install UMAP first: python -m pip install umap-learn') from e
    if len(z)<3: raise ValueError('UMAP requires at least three observations.')
    # Random init also works for small synthetic datasets; deterministic seed, single worker.
    model=umap.UMAP(n_components=2,n_neighbors=min(neighbors,len(z)-1),
                    min_dist=min_dist,metric=metric,random_state=seed,
                    transform_seed=seed,n_jobs=1,init='random')
    xy=np.asarray(model.fit_transform(np.asarray(z,dtype=np.float32)))
    return model,xy


def safe_transform(model, z):
    xy=np.asarray(model.transform(np.asarray(z,dtype=np.float32)))
    if xy.shape!=(len(z),2) or not np.isfinite(xy).all():
        raise RuntimeError('UMAP transform returned invalid coordinates.')
    return xy


def save_fig(fig,name,a):
    fig.savefig(a.outdir/'figures'/f'{name}.png',dpi=a.dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def title(fig, main, sub):
    fig.suptitle(main, fontsize=17, fontweight='bold', y=.985)
    fig.text(.5,.943,sub,ha='center',va='top',fontsize=10,color='#506074')


def axis2(ax, label='UMAP'):
    ax.set_xlabel(label+' 1'); ax.set_ylabel(label+' 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top','right']].set_visible(False)
    ax.set_aspect('equal',adjustable='box')


def points(ax, xy, values, labels, palette, s=6, alpha=.38, markers=None):
    for t in labels:
        m=values==t
        if m.any():
            ax.scatter(*xy[m].T,s=s,c=[palette[t]],alpha=alpha,linewidths=0,
                       marker='o' if markers is None else markers[t],rasterized=True)


def legend_items(labels,palette,markers=None):
    return [Line2D([],[],ls='',marker='o' if markers is None else markers[t],
                   color=palette[t],label=t,markersize=6) for t in labels]


def arrow(ax,start,end,color,width=1.5,alpha=.9,ls='-'):
    if np.linalg.norm(start-end)>1e-8:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,
                      linewidth=width,color=color,alpha=alpha,linestyle=ls,
                      shrinkA=4,shrinkB=4,zorder=5))
    ax.scatter(*start,s=28,marker='o',facecolors='white',edgecolors=[color],linewidths=1.0,zorder=6)
    ax.scatter(*end,s=32,marker='^',color=[color],edgecolors='white',linewidths=.4,zorder=6)



def cosine_cluster_order(vectors):
    """Average linkage on cosine distance of full-dimensional shift directions.

    Nonzero finite vectors are clustered; undefined directions are appended in
    their original order. The same permutation is used for both axes of each
    square cosine matrix. No clustering uses UMAP coordinates.
    """
    vectors=np.asarray(vectors,dtype=np.float64)
    good=np.isfinite(vectors).all(axis=1)&(np.linalg.norm(vectors,axis=1)>1e-12)
    valid=np.flatnonzero(good); invalid=np.flatnonzero(~good)
    if len(valid)>=2:
        u=unit(vectors[valid])
        dist=np.clip(1-u@u.T,0,2)
        dist=(dist+dist.T)*.5;np.fill_diagonal(dist,0.)
        tree=linkage(squareform(dist,checks=True),method='average',optimal_ordering=True)
        valid=valid[leaves_list(tree)]
    return np.concatenate([valid,invalid])


def direction_figures(meta,shifts,tc,co,a):
    """Observed source diagnostics: fixed-order and cosine-clustered heatmaps."""
    sources=sorted(meta.cell_type.unique())
    conditions=list(a.cytokines)
    d=shifts.shape[1]
    dims=[f'Z{i+1}' for i in range(d)]
    # Keep the same ordering in the tables, both axes and all panels.
    lookup={(r.cell_type,r.cytokine):i for i,r in enumerate(meta.itertuples(index=False))}
    order=[lookup[t,c] for c in conditions for t in sources]
    m=meta.iloc[order].reset_index(drop=True)
    V=shifts[order]
    valid=m.direction_defined.to_numpy(dtype=bool)
    U=unit(V); U[~valid]=np.nan
    sim=np.clip(U @ U.T,-1,1)
    labels=m.label.tolist()
    pd.DataFrame(sim,index=labels,columns=labels).to_csv(a.outdir/'shift_direction_cosine_similarity.csv',index_label='shift')
    pd.DataFrame(1-sim,index=labels,columns=labels).to_csv(a.outdir/'shift_direction_cosine_distance.csv',index_label='shift')
    pd.concat([m,pd.DataFrame(U,columns=dims)],axis=1).to_csv(a.outdir/'unit_latent_shift_vectors.csv',index=False)
    m.assign(display_order=np.arange(1,len(m)+1)).to_csv(a.outdir/'shift_display_order.csv',index=False)
    pd.DataFrame({'cell_type':sources}).to_csv(a.outdir/'cell_type_order.csv',index=False)
    pd.DataFrame({'cytokine':conditions}).to_csv(a.outdir/'cytokine_order.csv',index=False)
    # Direction UMAP is separate from cellular-state UMAP. Exclude undefined directions.
    if valid.sum()>=3:
        print(f'Embedding {valid.sum()} nonzero source shifts in {d}D (cosine UMAP)...',flush=True)
        _,xy=umap_fit(U[valid],a.shift_neighbors,a.min_dist,'cosine',a.seed)
        mv=m.loc[valid].reset_index(drop=True).copy()
        mv['UMAP1']=xy[:,0];mv['UMAP2']=xy[:,1]
        mv.to_csv(a.outdir/'shift_direction_umap.csv',index=False)
        fig,axs=plt.subplots(1,2,figsize=(14,7.5))
        for ax,field,ls,palette,lab in [(axs[0],'cytokine',conditions,co,'Colored by cytokine'),(axs[1],'cell_type',sources,tc,'Colored by source cell type')]:
            for t in ls:
                q=mv[field].eq(t).to_numpy()
                ax.scatter(*xy[q].T,s=53,color=[palette[t]],alpha=.9,edgecolors='white',linewidths=.5)
            axis2(ax,'Shift UMAP');ax.set_title(lab,fontweight='bold')
            ax.legend(handles=legend_items(ls,palette),loc='upper center',bbox_to_anchor=(.5,-.07),ncol=3,frameon=False,fontsize=9)
        title(fig,'UMAP of cytokine-induced latent-shift directions',f'One point = one source cell type × cytokine | {valid.sum()} unit vectors | cosine metric')
        fig.subplots_adjust(left=.055,right=.98,top=.86,bottom=.22,wspace=.15)
        save_fig(fig,'04_UMAP_shift_directions',a)
    else:
        print('Fewer than three nonzero shifts: skipping direction UMAP only.',flush=True)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#E4E7EB')
    # The original filenames retain the fixed-order version. A second version
    # uses the same global cosine cluster order for 05 (rows) and 06 (both axes).
    clustered_order=cosine_cluster_order(V)
    clustered_labels=[labels[i] for i in clustered_order]
    m.iloc[clustered_order].assign(cluster_order=np.arange(1,len(m)+1)).to_csv(
        a.outdir/'shift_cluster_order.csv',index=False)
    pd.DataFrame(sim[np.ix_(clustered_order,clustered_order)],index=clustered_labels,
                 columns=clustered_labels).to_csv(a.outdir/'shift_direction_cosine_similarity_clustered.csv',index_label='shift')
    pd.DataFrame(U[clustered_order],index=clustered_labels,columns=dims).to_csv(
        a.outdir/'unit_latent_shift_vectors_clustered.csv',index_label='shift')
    component_max=max(float(np.nanmax(np.abs(U))) if valid.any() else 0,1e-8)
    for clustered in (False,True):
        ii=clustered_order if clustered else np.arange(len(m))
        ls=[labels[i] for i in ii]
        suffix='_clustered' if clustered else ''
        for is_cos,A,name,heading,subtitle,vmax,cblab in [
            (False,U[ii],'05_latent_shift_directions','Signed components of measured perturbations',
             'Rows clustered by cosine distance | original latent dimension order' if clustered else 'Unit-normalized vectors | fixed cytokine and cell-type order',
             component_max,'Unit-vector component'),
            (True,sim[np.ix_(ii,ii)],'06_shift_cosine_similarity','Pairwise similarity of measured perturbation directions',
             'Identical row/column clustering | average linkage on cosine distance' if clustered else 'Fixed cytokine and cell-type order on both axes | no clustering',
             1.,'Cosine similarity')]:
            n=len(m); fig,ax=plt.subplots(figsize=(17 if is_cos else 15,max(8,n*.21+2)))
            im=ax.imshow(np.ma.masked_invalid(A),aspect='auto',cmap=cmap,vmin=-vmax,vmax=vmax,interpolation='nearest')
            ax.set_yticks(range(n),ls,fontsize=7)
            if is_cos:
                ax.set_xticks(range(n),ls,rotation=90,fontsize=6.5)
            else:
                ix=np.unique(np.linspace(0,d-1,9,dtype=int));ax.set_xticks(ix,[dims[i] for i in ix]);ax.set_xlabel('Original latent dimension')
            if not clustered:
                for i in range(len(sources),n,len(sources)):
                    ax.axhline(i-.5,color='white',lw=1.2)
                    if is_cos: ax.axvline(i-.5,color='white',lw=1.2)
            ax.tick_params(length=0)
            fig.colorbar(im,ax=ax,shrink=.65,pad=.02,label=cblab)
            title(fig,heading,subtitle+' | gray = undefined direction')
            fig.subplots_adjust(left=.20,right=.96,bottom=.22 if is_cos else .08,top=.91)
            save_fig(fig,name+suffix,a)
    # Fixed-order panels retain the same order for every cytokine/cell type.
    # Clustered panels independently order their own vectors by cosine distance.
    folder=a.outdir/'group_cosine_matrices';folder.mkdir(exist_ok=True)
    orders=[]
    for group_field,item_field,groups,ls,ncols,name,heading in [
        ('cytokine','cell_type',conditions,sources,min(3,len(conditions)),'07_cell_type_shift_similarity_per_cytokine','Cell-type response similarity for each cytokine'),
        ('cell_type','cytokine',sources,conditions,min(4,len(sources)),'08_cytokine_shift_similarity_per_cell_type','Cytokine response similarity for each cell type')]:
        nr=int(np.ceil(len(groups)/ncols)); n=len(ls)
        for clustered in (False,True):
            suffix='_clustered' if clustered else ''
            fig,axs=plt.subplots(nr,ncols,figsize=(5.4*ncols,5.0*nr+1.1),squeeze=False,layout='constrained')
            for ax,g in zip(axs.flat,groups):
                ids=[lookup[t,c] for t,c in ([(t,g) for t in ls] if group_field=='cytokine' else [(g,c) for c in ls])]
                vv=shifts[ids]; uu=unit(vv); uu[np.linalg.norm(vv,axis=1)<=1e-12]=np.nan
                order=cosine_cluster_order(vv) if clustered else np.arange(n)
                panel_labels=[ls[i] for i in order]
                val=np.clip(uu@uu.T,-1,1)[np.ix_(order,order)]
                im=ax.imshow(np.ma.masked_invalid(val),cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
                ax.set_xticks(range(n),panel_labels,rotation=65,ha='right',fontsize=8)
                ax.set_yticks(range(n),panel_labels,fontsize=8)
                ax.tick_params(length=0);ax.set_title(g,fontweight='bold',fontsize=12)
                for sp in ax.spines.values():sp.set_visible(False)
                if n<=14:
                    for i in range(n):
                        for j in range(n):
                            v=val[i,j];ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=6 if n>8 else 9,color='white' if np.isfinite(v) and abs(v)>.65 else '#202630')
                pd.DataFrame(val,index=panel_labels,columns=panel_labels).to_csv(
                    folder/f'{group_field}__{slug(g)}{suffix}.csv',index_label=item_field)
                orders.extend(dict(group_field=group_field,group=g,version='clustered' if clustered else 'fixed',position=i+1,label=label)
                              for i,label in enumerate(panel_labels))
            for ax in list(axs.flat)[len(groups):]:ax.axis('off')
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1,1),cmap=cmap),ax=list(axs.flat)[:len(groups)],shrink=.62,pad=.025,label='Cosine similarity')
            subtitle='Each panel clustered independently | same row/column permutation' if clustered else 'Same row and column order in every panel | no clustering'
            fig.suptitle(heading+'\n'+subtitle+' | full 200D vectors',fontsize=16,fontweight='bold')
            save_fig(fig,name+suffix,a)
    pd.DataFrame(orders).to_csv(a.outdir/'group_cosine_display_orders.csv',index=False)
    pivot=meta.pivot(index='cell_type',columns='cytokine',values='norm').reindex(index=sources,columns=conditions)
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(pivot,cmap='viridis',aspect='auto')
    ax.set_xticks(range(len(conditions)),conditions,rotation=35,ha='right');ax.set_yticks(range(len(sources)),sources)
    for i in range(len(sources)):
        for j in range(len(conditions)):
            v=pivot.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if np.isfinite(v) and im.norm(v)<.52 else '#202630')
    fig.colorbar(im,ax=ax,label='L2 norm of latent shift');ax.set_title('Magnitude of cytokine-specific latent shifts',fontweight='bold',pad=14)
    fig.tight_layout();save_fig(fig,'09_latent_shift_magnitudes',a)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=Path('data/hcd_subset.h5ad'))
    p.add_argument('--shared-z200',type=Path,default=Path('data/shared_CD14_Mono_Z200'))
    p.add_argument('--heads-dir',type=Path,default=Path('data/nonlinear_shared_Z200_E32_fixed100'))
    p.add_argument('--holdout',default='CD14 Mono')
    p.add_argument('--cytokines',nargs='+',default=CYTOKINES,help='Must match the cytokines in the trained checkpoints.')
    p.add_argument('--heads',nargs='+',choices=HEADS,default=HEADS,help='Only hybrid is supported by this comparison.')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--keep-rare',action='store_true',help='Use only if the checkpoints were trained with rare cell types retained.')
    p.add_argument('--no-observed-target',action='store_true',help='Do not encode or display measured target treated cells.')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--n-neighbors',type=int,default=30)
    p.add_argument('--shift-neighbors',type=int,default=10)
    p.add_argument('--min-dist',type=float,default=.25)
    p.add_argument('--state-metric',choices=['euclidean','cosine'],default='euclidean')
    p.add_argument('--max-fit-per-group',type=int,default=0,help='0 fits UMAP on every reference cell. Positive cap affects visualization only.')
    p.add_argument('--max-points-per-group',type=int,default=300,help='Display cap, 0 displays every cell. Never changes full-dimensional centroids.')
    p.add_argument('--skip-direction-details',action='store_true',help='Skip the measured source-direction UMAP and heatmaps.')
    p.add_argument('--skip-source-model-panels',action='store_true',help='Skip the in-sample source prediction panels; target predictions are retained.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--formats',nargs='+',choices=['png'],default=['png'],help='PNG only (optional argument; retained for compatibility).')
    p.add_argument('--outdir',type=Path,default=Path('data/all_latent_shifts_Z200_hybrid'))
    a=p.parse_args()
    if ('PBS' in a.cytokines or len(set(a.cytokines))!=len(a.cytokines) or
        len(set(a.heads))!=len(a.heads) or
        min(a.n_neighbors,a.shift_neighbors)<2 or a.batch_size<1 or a.dpi<72 or
        min(a.max_fit_per_group,a.max_points_per_group,a.seed)<0 or not 0<=a.min_dist<=1):
        p.error('Invalid cytokines, heads, sampling or UMAP options.')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; use --device cpu.')
    for src in [a.shared_z200,a.heads_dir]:
        if src.resolve()==a.outdir.resolve() or src.resolve() in a.outdir.resolve().parents:
            p.error('Output must be outside the trained run directories.')
    if a.outdir.exists() and (not a.outdir.is_dir() or any(a.outdir.iterdir())):
        p.error('Choose a new or empty output directory.')
    return a


def load_input(a):
    import anndata as ad
    import scgen
    from scgen._scgenvae import SCGENVAE
    # Apply the same compatibility patch as in the training script.
    orig=SCGENVAE.inference
    def inference_fixed(self,x):
        o=orig(self,x)
        if 'qz_m' in o: o.setdefault('qzm',o['qz_m'])
        if 'qz_v' in o: o.setdefault('qzv',o['qz_v'])
        return o
    SCGENVAE.inference=inference_fixed
    data=ad.read_h5ad(a.data)
    data=data[data.obs.cytokine.isin(['PBS']+a.cytokines)].copy()
    if not a.keep_rare: data=data[~data.obs.cell_type.isin(RARE)].copy()
    if 'counts' in data.layers: del data.layers['counts']
    data.obs['condition']=data.obs.cytokine.astype(str)
    data.obs['cell_type']=data.obs.cell_type.astype(str)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError('Cell and gene names must be unique.')
    ct=data.obs.cell_type.to_numpy(); cc=data.obs.condition.to_numpy()
    if a.holdout not in set(ct): raise ValueError(f'Missing target {a.holdout}')
    sources=sorted(set(ct)-{a.holdout})
    if not sources: raise ValueError('No source types.')
    for t in sources+[a.holdout]:
        for c in ['PBS']+a.cytokines:
            if t==a.holdout and c!='PBS' and a.no_observed_target: continue
            if not np.any((ct==t)&(cc==c)): raise ValueError(f'Missing group {t}/{c}')
    held=(ct==a.holdout)&(cc!='PBS')
    ref=data[~held].copy(); ref.X=dense(ref.X).astype(np.float32,copy=False)
    # Query held separately, never provided to model loading or the reference fit.
    query=None if a.no_observed_target else data[held].copy()
    if query is not None: query.X=dense(query.X).astype(np.float32,copy=False)
    del data
    return scgen,ref,query,sources


def read_saved_shifts(run,d,a):
    tag=a.holdout.replace(' ','_').replace('/','-')
    out={}
    for c in a.cytokines:
        path=run/f'{c}__{tag}_latent_shift.npy'
        if not path.is_file():
            raise FileNotFoundError(f'{path} missing. Run the corresponding shared scGen benchmark first. Do not substitute a different centroid estimator.')
        v=np.asarray(np.load(path,allow_pickle=False),dtype=np.float32).reshape(-1)
        if v.shape!=(d,) or not np.isfinite(v).all(): raise ValueError(f'Invalid saved shift: {path}, shape {v.shape}')
        out[c]=v
    return out


def checkpoint_load(path):
    if not path.is_file(): raise FileNotFoundError(f'Missing trained head: {path}')
    try: return torch.load(path,map_location='cpu',weights_only=True)
    except TypeError: return torch.load(path,map_location='cpu') # older torch; trusted local training checkpoint only


def load_heads(a,genes,sources,digest):
    heads={}
    for kind in a.heads:
        path=a.heads_dir/f'head_{kind}.pt'; ck=checkpoint_load(path)
        if ck['kind']!=kind or ck['latent_dim']!=200 or ck['holdout']!=a.holdout:
            raise ValueError(f'Head kind/dimension/holdout mismatch: {path}')
        if list(ck['genes'])!=list(genes) or sorted(ck['source_types'])!=sorted(sources):
            raise ValueError(f'Gene order or source cell types do not match: {path}')
        if set(ck['cytokines'])!=set(a.cytokines) or len(ck['cytokines'])!=len(a.cytokines):
            raise ValueError(f'Cytokine set mismatch: {path}')
        if ck.get('vae_state_sha256')!=digest:
            raise ValueError(f'{path} was not trained on this exact Z200 VAE (state hash mismatch). Use its matching checkpoint.')
        h=PerturbationHead(200,len(ck['cytokines']),ck['embedding_dim'],ck['hidden'],ck['dropout'],kind)
        h.load_state_dict(ck['state_dict'],strict=True); h.to(a.device).eval().requires_grad_(False)
        # Map labels according to the actual embedding row order stored with the head.
        heads[kind]=(h,{c:i for i,c in enumerate(ck['cytokines'])})
        print(f'Loaded {kind}: {sum(p.numel() for p in h.parameters()):,} head parameters; VAE hash verified.',flush=True)
    return heads


def encode(scgen,ref,query,run,d,a,sources):
    path=run/('shared__'+a.holdout.replace(' ','_').replace('/','-'))
    if not (path/'model.pt').is_file(): raise FileNotFoundError(f'Missing VAE checkpoint: {path}')
    print(f'Loading Z{d}: {path} | {ref.n_obs:,} reference cells',flush=True)
    model=scgen.SCGEN.load(str(path),adata=ref,accelerator='cpu',device='auto')
    model.to_device(a.device); model.module.eval().requires_grad_(False)
    digest=model_digest(model.module)
    with torch.inference_mode():
        zr=np.asarray(model.get_latent_representation(give_mean=True,batch_size=a.batch_size),dtype=np.float32)
    if zr.shape!=(ref.n_obs,d) or not np.isfinite(zr).all(): raise ValueError(f'Invalid Z{d} latent matrix: {zr.shape}')
    # Compute every model prediction from reference PBS and saved weights/shifts first.
    saved=read_saved_shifts(run,d,a)
    heads=load_heads(a,list(ref.var_names.astype(str)),sources,digest)
    obs_ref=pd.DataFrame({'cell_id':ref.obs_names.astype(str),'cell_type':ref.obs.cell_type.to_numpy(),
                           'cytokine':ref.obs.condition.to_numpy(),'status':'reference'}).reset_index(drop=True)
    pred,source_pred=compute_predictions(zr,obs_ref,saved,heads,a)
    del heads
    # Encode observed target only after all predictions are already fixed.
    obs=obs_ref; z=zr
    if query is not None:
        with torch.inference_mode():
            zq=np.asarray(model.get_latent_representation(query,give_mean=True,batch_size=a.batch_size),dtype=np.float32)
        if zq.shape!=(query.n_obs,d) or not np.isfinite(zq).all(): raise ValueError(f'Invalid target treated embedding: {zq.shape}')
        oq=pd.DataFrame({'cell_id':query.obs_names.astype(str),'cell_type':query.obs.cell_type.to_numpy(),
                         'cytokine':query.obs.condition.to_numpy(),'status':'heldout_observed'}).reset_index(drop=True)
        obs=pd.concat([obs_ref,oq],ignore_index=True); z=np.concatenate([zr,zq])
    if digest!=model_digest(model.module): raise RuntimeError('VAE state changed during visualization.')
    del model; gc.collect()
    if a.device=='cuda': torch.cuda.empty_cache()
    return z,obs,pred,source_pred,digest


def head_predict(z,head,ci,a):
    h,ids=head
    parts=[]
    with torch.inference_mode():
        for i in range(0,len(z),a.batch_size):
            q=torch.as_tensor(z[i:i+a.batch_size],dtype=torch.float32,device=a.device)
            parts.append(h(q,ids[ci]).cpu().numpy())
    return np.concatenate(parts)


def compute_predictions(z,obs,saved,heads,a):
    ct=obs.cell_type.to_numpy(); cc=obs.cytokine.to_numpy()
    if np.any((ct==a.holdout)&(cc!='PBS')): raise ValueError('Target treated cells in reference prediction input.')
    pbs={t:z[(ct==t)&(cc=='PBS')] for t in sorted(set(ct))}
    for t,v in pbs.items():
        if not len(v): raise ValueError(f'Missing PBS: {t}')
    target=pbs[a.holdout]; preds={}; source={}
    for c in a.cytokines:
        for kind in ['default']+list(heads):
            for t,v in pbs.items():
                if a.skip_source_model_panels and t!=a.holdout: continue
                pp=v+saved[c] if kind=='default' else head_predict(v,heads[kind],c,a)
                if pp.shape!=v.shape or not np.isfinite(pp).all(): raise ValueError(f'Invalid predicted latent states: {kind}/{t}/{c}')
                if t==a.holdout: preds[kind,c]=pp
                else:
                    # Only group centroids and within-type shift variability required for source diagnostics.
                    dv=pp-v; dm=dv.mean(0,dtype=np.float64)
                    source[kind,t,c]=(pp.mean(0,dtype=np.float64),float(np.sqrt(np.mean(np.sum((dv-dm)**2,axis=1)))))
    return preds,source


def reference_stats(z,obs,a):
    ref=obs.status.eq('reference').to_numpy(); ct=obs.cell_type.to_numpy(); cc=obs.cytokine.to_numpy()
    types=sorted(set(ct[ref])); sources=[t for t in types if t!=a.holdout]
    centers={}; counts={}
    for t in types:
        for c in ['PBS']+a.cytokines:
            mask=ref&(ct==t)&(cc==c)
            if t==a.holdout and c!='PBS': continue
            if not mask.any(): raise ValueError(f'Missing reference {t}/{c}')
            centers[t,c]=z[mask].mean(0,dtype=np.float64); counts[t,c]=int(mask.sum())
    meta=[]; shifts=[]
    for c in a.cytokines:
        for t in sources:
            v=centers[t,c]-centers[t,'PBS']; shifts.append(v)
            norm=float(np.linalg.norm(v))
            meta.append(dict(cell_type=t,cytokine=c,label=f'{c} | {t}',n_PBS=counts[t,'PBS'],n_treated=counts[t,c],norm=norm,direction_defined=norm>1e-12))
    return centers,counts,pd.DataFrame(meta),np.stack(shifts),types,sources


def cosine_pair(u,v):
    a=np.linalg.norm(u); b=np.linalg.norm(v)
    return float(np.clip(np.dot(u,v)/(a*b),-1,1)) if min(a,b)>1e-12 else np.nan


def numeric_tables(z,obs,preds,source,centers,counts,meta,shifts,d,a):
    dims=[f'Z{i+1}' for i in range(d)]; out=a.outdir
    np.savez_compressed(out/'latent_embeddings.npz',z=z,**{k:obs[k].to_numpy(dtype=str) for k in obs.columns})
    rows=[]
    for (t,c,status),ids in obs.groupby(['cell_type','cytokine','status']).indices.items():
        rows.append(dict(cell_type=t,cytokine=c,status=status,n_cells=len(ids),**dict(zip(dims,z[ids].mean(0,dtype=np.float64)))))
    pd.DataFrame(rows).to_csv(out/'latent_centroids.csv',index=False)
    pd.concat([meta,pd.DataFrame(shifts,columns=dims)],axis=1).to_csv(out/'raw_latent_shift_vectors.csv',index=False)
    ref=obs.status.eq('reference').to_numpy()
    pd.DataFrame({'dimension':dims,'variance':z[ref].var(0)}).to_csv(out/'latent_dimension_variance.csv',index=False)
    ct=obs.cell_type.to_numpy(); cc=obs.cytokine.to_numpy()
    pbs=z[ref&(ct==a.holdout)&(cc=='PBS')]; base=centers[a.holdout,'PBS']; rows=[]; vectors=[]
    truth={c:z[(~ref)&(ct==a.holdout)&(cc==c)] for c in a.cytokines}
    for (kind,c),pp in preds.items():
        delta=pp-pbs; v=pp.mean(0,dtype=np.float64)-base
        obs_v=truth[c].mean(0,dtype=np.float64)-base if len(truth[c]) else None
        rows.append(metric_row(kind,a.holdout,c,'heldout_target',v,obs_v,float(np.sqrt(np.mean(np.sum((delta-delta.mean(0,dtype=np.float64))**2,axis=1)))),d))
        vectors.append(dict(model=kind,cell_type=a.holdout,cytokine=c,role='heldout_prediction',**dict(zip(dims,v))))
    for (kind,t,c),(m,rms) in source.items():
        v=m-centers[t,'PBS']; obs_v=centers[t,c]-centers[t,'PBS']
        rows.append(metric_row(kind,t,c,'source_in_sample',v,obs_v,rms,d))
        vectors.append(dict(model=kind,cell_type=t,cytokine=c,role='source_prediction',**dict(zip(dims,v))))
    for c,v in truth.items():
        if len(v): vectors.append(dict(model='observed',cell_type=a.holdout,cytokine=c,role='heldout_observed',**dict(zip(dims,v.mean(0,dtype=np.float64)-base))))
    metrics=pd.DataFrame(rows); metrics.to_csv(out/'latent_prediction_metrics.csv',index=False)
    pd.DataFrame(vectors).to_csv(out/'predicted_and_observed_mean_shifts.csv',index=False)
    # Save paired computational predictions; no pairing with observed treated cells is implied.
    keys=list(preds)
    np.savez_compressed(out/'target_predicted_latent_states.npz',model=np.array([k for k,c in keys]),cytokine=np.array([c for k,c in keys]),
                       pbs_latent=pbs,predicted_latent=np.stack([preds[k] for k in keys]),
                       cell_id=obs.loc[ref&(ct==a.holdout)&(cc=='PBS'),'cell_id'].to_numpy(dtype=str))
    return metrics


def metric_row(kind,t,c,role,v,truth,rms,d):
    n=float(np.linalg.norm(v)); tn=float(np.linalg.norm(truth)) if truth is not None else np.nan
    err=float(np.linalg.norm(v-truth)) if truth is not None else np.nan
    return dict(latent_dim=d,model=kind,cell_type=t,cytokine=c,role=role,
                predicted_mean_shift_norm=n,observed_mean_shift_norm=tn,
                cosine_to_observed=cosine_pair(v,truth) if truth is not None else np.nan,
                relative_centroid_error=err/tn if tn>1e-12 else np.nan,
                predicted_to_observed_norm_ratio=n/tn if tn>1e-12 else np.nan,
                latent_centroid_rmse=err/np.sqrt(d),within_type_shift_rms=rms)


def fit_projection(z,obs,a):
    ref=obs.status.eq('reference').to_numpy()
    used=stratified_ids(obs,ref,a.max_fit_per_group,a.seed)
    print(f'Fitting Z200 reference UMAP on {len(used):,} cells (no held-out treated or predicted cells).',flush=True)
    model,xy=umap_fit(z[used],a.n_neighbors,a.min_dist,a.state_metric,a.seed)
    transform=lambda v:safe_transform(model,v)
    coords=np.empty((len(z),2),dtype=np.float32); coords[used]=xy
    missing=np.setdiff1d(np.arange(len(z)),used)
    for i in range(0,len(missing),4096):
        k=missing[i:i+4096]; coords[k]=transform(z[k])
    if not np.isfinite(coords).all(): raise ValueError('Nonfinite UMAP coordinates')
    return coords,transform,used,['UMAP 1','UMAP 2']


def bounds(*arrs):
    vs=[np.asarray(x).reshape(-1,2) for x in arrs if np.asarray(x).size]
    v=np.concatenate(vs); lo=v.min(0); hi=v.max(0)
    pad=max(float(np.max(hi-lo))*.075,.05)
    return ((lo[0]-pad,hi[0]+pad),(lo[1]-pad,hi[1]+pad))


def setup(ax,b,labels):
    ax.set_xlim(*b[0]); ax.set_ylim(*b[1]); ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel(labels[0],fontsize=9); ax.set_ylabel(labels[1],fontsize=9)
    if 'UMAP' in labels[0]: ax.set_xticks([]); ax.set_yticks([])
    else: ax.tick_params(labelsize=8); ax.grid(alpha=.12)
    ax.spines[['top','right']].set_visible(False)


def cloud(ax,x,color,alpha=.33,size=7,marker='o'):
    if len(x): ax.scatter(*x.T,c=[color],s=size,alpha=alpha,marker=marker,linewidths=0,rasterized=True,zorder=2)


def draw_arrow(ax,start,end,color,style='-',lw=1.8,marker='D',draw_start=False):
    if np.linalg.norm(start-end)>1e-8:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,color=color,
                                     linewidth=lw,linestyle=style,shrinkA=3,shrinkB=3,zorder=6))
    if draw_start: ax.scatter(*start,s=35,marker='o',facecolors='white',edgecolors=[color],linewidths=1.1,zorder=7)
    ax.scatter(*end,s=43,marker=marker,c=[color],edgecolors='white',linewidths=.55,zorder=8)


def model_name(k,d): return f'Shared scGen Z{d}' if k=='default' else MODEL_NAMES[k]


def panels(n,ncols=3,w=4.7,h=4.0):
    nc=min(n,ncols); nr=int(np.ceil(n/nc))
    fig,axs=plt.subplots(nr,nc,figsize=(nc*w,nr*h+1.0),squeeze=False)
    for ax in list(axs.flat)[n:]: ax.axis('off')
    return fig,axs


def layout(fig,bottom=.15,top=.85):
    fig.subplots_adjust(left=.07,right=.98,top=top,bottom=bottom,wspace=.26,hspace=.36)


def marker_handle(c,label,marker='o',ls='None'):
    return Line2D([],[],color=c,label=label,marker=marker,ls=ls,markersize=6,linewidth=1.7)


def base_legend(fig,handles,ncol=4):
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=min(ncol,len(handles)),frameon=False,fontsize=9)


def projection_figures(z,obs,preds,source,centers,types,sources,metrics,d,a,tc,co):
    xy,transform,used,labels=fit_projection(z,obs,a)
    mode="UMAP"
    ref=obs.status.eq('reference').to_numpy(); ct=obs.cell_type.to_numpy(); cc=obs.cytokine.to_numpy()
    draw=stratified_ids(obs,ref,a.max_points_per_group,a.seed+1)
    target_mask=ref&(ct==a.holdout)&(cc=='PBS'); target=xy[target_mask]; base=centers[a.holdout,'PBS']
    # Project full-dimensional endpoints separately from cell clouds.
    ek=list(centers); ev=list(centers.values())
    for (k,c),v in preds.items(): ek.append(('prediction',k,c)); ev.append(v.mean(0,dtype=np.float64))
    for (k,t,c),(v,_) in source.items(): ek.append(('source_prediction',k,t,c)); ev.append(v)
    # Fix all predicted endpoints before transforming observed target centroids.
    ep=dict(zip(ek,transform(np.stack(ev))))
    truth={c:xy[(~ref)&(ct==a.holdout)&(cc==c)] for c in a.cytokines}
    ok=[]; ov=[]
    for c in a.cytokines:
        m=(~ref)&(ct==a.holdout)&(cc==c)
        if m.any(): ok.append(('observed',c)); ov.append(z[m].mean(0,dtype=np.float64))
    if ov: ep.update(zip(ok,transform(np.stack(ov))))
    pxy={k:transform(v) for k,v in preds.items()}
    if not all(np.isfinite(v).all() for v in list(ep.values())+list(pxy.values())): raise ValueError('Invalid predicted projected states.')
    prefix=mode.lower(); folder=a.outdir/'figures'/prefix; folder.mkdir(parents=True,exist_ok=True)
    # save_fig expects outdir/figures; use a small config view to target each projection folder.
    def save(fig,name):
        fig.savefig(folder/f'{name}.png',dpi=a.dpi,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    coords=obs.copy(); coords['axis1']=xy[:,0]; coords['axis2']=xy[:,1]; coords['used_for_projection_fit']=np.isin(np.arange(len(z)),used)
    coords.to_csv(a.outdir/f'{prefix}_cell_coordinates.csv',index=False)
    pd.DataFrame([dict(endpoint=' | '.join(k),axis1=v[0],axis2=v[1]) for k,v in ep.items()]).to_csv(a.outdir/f'{prefix}_centroid_endpoints.csv',index=False)
    keys=list(pxy); np.savez_compressed(a.outdir/f'{prefix}_target_predictions.npz',model=np.array([k for k,c in keys]),cytokine=np.array([c for k,c in keys]),xy=np.stack([pxy[k] for k in keys]),pbs_xy=target,cell_id=obs.loc[target_mask,'cell_id'].to_numpy(dtype=str))
    rb=bounds(xy[ref],np.stack([ep[k] for k in centers])); tb=bounds(target,*truth.values(),*pxy.values(),np.stack([ep[k] for k in ep if k[0] in ('prediction','observed')]),ep[a.holdout,'PBS'])
    # Overview: one fitted representation, two different color annotations.
    fig,axs=plt.subplots(1,2,figsize=(14,7.7))
    for ax,values,labs,palette,label in [(axs[0],ct,types,tc,'Cell type'),(axs[1],cc,['PBS']+a.cytokines,co,'Treatment')]:
        points(ax,xy[draw],values[draw],labs,palette,s=8,alpha=.47); setup(ax,rb,labels); ax.set_title(label,fontweight='bold')
        ax.legend(handles=legend_items(labs,palette),loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False,fontsize=8.5)
    title(fig,f'Shared Z{d}: cellular states',f'UMAP fitted only on reference cells using all {d} latent dimensions')
    layout(fig,bottom=.25,top=.85);save(fig,'01_cell_state')
    # Per-treatment maps: directly inspect observed source distributions in same axes.
    fig,axs=panels(len(a.cytokines)+1,ncols=4,w=3.65,h=3.6)
    for ax,c in zip(axs.flat,['PBS']+a.cytokines):
        cloud(ax,xy[draw],'#CDD3DA',.11,3); ids=draw[cc[draw]==c]; points(ax,xy[ids],ct[ids],types,tc,s=8,alpha=.6)
        setup(ax,rb,labels);ax.set_title(c,fontweight='bold')
    title(fig,f'Shared Z{d}: cellular states by treatment',f'{mode} | same reference coordinates and limits; target treated cells excluded')
    base_legend(fig,legend_items(types,tc),5);layout(fig,bottom=.18);save(fig,'02_states_by_treatment')
    # Measured source perturbations: PBS -> observed treated centroid, not predicted.
    fig,axs=panels(len(a.cytokines))
    for ax,c in zip(axs.flat,a.cytokines):
        cloud(ax,xy[draw],'#C1CAD3',.17,4)
        for t in sources:
            draw_arrow(ax,ep[t,'PBS'],ep[t,c],tc[t],lw=1.65,marker='^',draw_start=True)
        ax.scatter(*ep[a.holdout,'PBS'],s=85,marker='*',c='#222A33',edgecolors='white',linewidths=.6,zorder=9)
        setup(ax,rb,labels); ax.set_title(c,fontweight='bold')
    hs=legend_items(sources,tc)+[marker_handle('#333333','PBS centroid','o'),marker_handle('#333333','Measured treated centroid','^'),marker_handle('#222A33',f'{a.holdout} PBS','*')]
    title(fig,f'Z{d}: measured source-cell perturbation arrows',f'{mode} | Each arrow connects experimentally measured PBS and treated latent centroids')
    base_legend(fig,hs,5);layout(fig,bottom=.18);save(fig,'03_source_observed_arrows')
    # Held-out target movement. All model predictions on identical axes within this VAE.
    kinds=list(dict.fromkeys(k for k,c in preds)); start=ep[a.holdout,'PBS']
    target_h=[marker_handle(PBS_COLOR,'PBS cells / starting centroid','o'),marker_handle(OBS_COLOR,'Observed treated centroid','^','--')]+[marker_handle(MODEL_COLORS[k],model_name(k,d),'D','-') for k in kinds]
    if not ok: target_h.pop(1)
    fig,axs=panels(len(a.cytokines))
    for ax,c in zip(axs.flat,a.cytokines):
        cloud(ax,xy[draw],'#D1D8DF',.10,3)
        cloud(ax,target,PBS_COLOR,.42,9);cloud(ax,truth[c],OBS_COLOR,.15,8,'^')
        if ('observed',c) in ep: draw_arrow(ax,start,ep['observed',c],OBS_COLOR,'--',2.0,'^')
        for k in kinds: draw_arrow(ax,start,ep['prediction',k,c],MODEL_COLORS[k],lw=2.0)
        ax.scatter(*start,s=72,c=PBS_COLOR,edgecolors='white',linewidths=.7,zorder=9)
        setup(ax,tb,labels);ax.set_title(c,fontweight='bold')
    title(fig,f'Z{d}: {a.holdout} predicted perturbation endpoints',f'{mode} | Solid = predicted from PBS; dashed = experimentally observed | common axes within Z{d}')
    base_legend(fig,target_h,4);layout(fig,bottom=.14);save(fig,'04_target_all_model_arrows')
    # No target-only panels on the reference-fitted map. The separate target-only
    # UMAP below provides the detailed target comparison and predicted clouds.
    # Source model panels are descriptive in-sample checks, not generalization tests.
    if mode=='UMAP' and source:
        sb=bounds(xy[ref],np.stack([ep[key] for key in ep if key[0]=='source_prediction']))
        for k in kinds:
            fig,axs=panels(len(a.cytokines))
            for ax,c in zip(axs.flat,a.cytokines):
                cloud(ax,xy[draw],'#C1CAD3',.14,4)
                for t in sources:
                    p=ep[t,'PBS']; q=ep[t,c]; r=ep['source_prediction',k,t,c]
                    draw_arrow(ax,p,q,tc[t],'--',1.0,'^');draw_arrow(ax,p,r,tc[t],'-',1.7,'D',True)
                setup(ax,sb,labels);ax.set_title(c,fontweight='bold')
            title(fig,f'Z{d}: {model_name(k,d)} — source-cell predictions', 'In-sample diagnostic | solid: predicted endpoint; dashed: measured treated endpoint')
            base_legend(fig,legend_items(sources,tc)+[marker_handle('#333333','Predicted','D','-'),marker_handle('#333333','Observed','^','--')],5);layout(fig,bottom=.17)
            save(fig,f'05_source_predictions_{k}')
    return xy,ep,pxy,tb,labels


def target_umap_overview(xy,cc,preds_xy,endpoints,kinds,folder,a):
    """Six overview figures on ONE measured-target-fitted UMAP.

    Match the reference-map figure sequence, but every point belongs to the
    held-out cell type. The two model-specific summaries show predicted clouds
    as well as observed and predicted centroid movements. All overview panels
    share exactly the same limits; the separate single-cytokine figures zoom in.
    """
    folder=folder/'overview';folder.mkdir(exist_ok=True)
    labels=['Target UMAP 1','Target UMAP 2']
    pbs=xy[cc=='PBS'];start=endpoints['PBS','PBS']
    conditions=[c for c in ['PBS']+a.cytokines if (cc==c).any()]
    observed=any(c!='PBS' for c in conditions)
    colors={c:COND_COLORS.get(c,'#A0A0A0') for c in conditions}
    markers={'default':'D','hybrid':'h'}
    bb=bounds(xy,*preds_xy.values(),np.stack(list(endpoints.values())))
    scope='One UMAP fitted on measured target PBS and treated cells; retrospective display' if observed else 'One UMAP fitted on measured target PBS cells only'
    def save(fig,name):
        fig.savefig(folder/f'{name}.png',dpi=a.dpi,bbox_inches='tight',facecolor='white');plt.close(fig)
    def start_mark(ax):
        ax.scatter(*start,s=75,marker='o',facecolors='white',edgecolors='#303C49',linewidths=1.5,zorder=10)
    pbs_handle=marker_handle(PBS_COLOR,'PBS cells')
    start_handle=Line2D([],[],ls='',marker='o',mfc='white',mec='#303C49',mew=1.5,label='PBS centroid',markersize=7)
    obs_handles=[marker_handle(OBS_COLOR,'Observed treated cells','^'),marker_handle(OBS_COLOR,'Observed centroid','^','--')] if observed else []
    model_handles={k:marker_handle(MODEL_COLORS[k],model_name(k,200),markers[k],'-') for k in kinds}
    # 01: the target cell type is constant, so show control vs treatment instead
    # of a redundant single-color cell-type panel.
    fig,axs=plt.subplots(1,2,figsize=(12.5,6.8))
    ax=axs[0];cloud(ax,pbs,PBS_COLOR,.60,12);cloud(ax,xy[cc!='PBS'],OBS_COLOR,.30,12,'^');setup(ax,bb,labels);ax.set_title('PBS vs treated',fontweight='bold')
    ax.legend(handles=[pbs_handle]+([marker_handle(OBS_COLOR,'All measured treated cells','^')] if observed else []),loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,frameon=False,fontsize=9)
    ax=axs[1];points(ax,xy,cc,conditions,colors,s=12,alpha=.62);setup(ax,bb,labels);ax.set_title('Treatment',fontweight='bold')
    ax.legend(handles=legend_items(conditions,colors),loc='upper center',bbox_to_anchor=(.5,-.10),ncol=3,frameon=False,fontsize=9)
    title(fig,f'{a.holdout}: cellular states',scope);layout(fig,bottom=.24);save(fig,'01_cell_state')
    # 02: highlight the selected measured condition; other measured target cells
    # remain a very light contextual backdrop, never source-cell types.
    fig,axs=panels(len(conditions),ncols=4,w=3.9,h=3.7)
    for ax,c in zip(axs.flat,conditions):
        cloud(ax,xy[cc!=c],'#D3D8DE',.17,8);cloud(ax,xy[cc==c],colors[c],.70,12)
        setup(ax,bb,labels);ax.set_title(f'{c} (n={(cc==c).sum():,})',fontweight='bold')
    title(fig,f'{a.holdout}: cellular states by treatment','Measured target cells only | same UMAP and axis limits in every panel')
    base_legend(fig,legend_items(conditions,colors),4);layout(fig,bottom=.15);save(fig,'02_states_by_treatment')
    def measured(ax,c,show_arrow=True):
        cloud(ax,pbs,PBS_COLOR,.45,12);cloud(ax,xy[cc==c],OBS_COLOR,.38,13,'^')
        if show_arrow and ('observed',c) in endpoints:
            draw_arrow(ax,start,endpoints['observed',c],OBS_COLOR,'--',2.4,'^')
    # 03: observed movement only; no model prediction or source cells.
    if observed:
        fig,axs=panels(len(a.cytokines),w=4.8,h=4.2)
        for ax,c in zip(axs.flat,a.cytokines):
            measured(ax,c);start_mark(ax);setup(ax,bb,labels);ax.set_title(c,fontweight='bold')
        title(fig,f'{a.holdout}: measured perturbation arrows','PBS to observed treated centroid | common target-only UMAP')
        base_legend(fig,[pbs_handle,start_handle]+obs_handles,4);layout(fig,bottom=.13);save(fig,'03_observed_target_arrows')
    # 04: all model endpoints in each cytokine panel; same map as 03.
    fig,axs=panels(len(a.cytokines),w=4.8,h=4.2)
    for ax,c in zip(axs.flat,a.cytokines):
        measured(ax,c)
        for k in kinds:draw_arrow(ax,start,endpoints[k,c],MODEL_COLORS[k],'-',2.3,markers[k])
        start_mark(ax);setup(ax,bb,labels);ax.set_title(c,fontweight='bold')
    title(fig,f'{a.holdout}: predicted perturbation endpoints','Solid = predicted; dashed = measured | common target-only UMAP' if observed else 'Solid = predicted from PBS | common target-only UMAP')
    base_legend(fig,[pbs_handle,start_handle]+obs_handles+list(model_handles.values()),3);layout(fig,bottom=.15);save(fig,'04_target_all_model_arrows')
    # 05: directly comparable full distributions and centroid endpoints, one
    # six-panel figure per model. Predictions are pre-decoder latent states.
    for k in kinds:
        fig,axs=panels(len(a.cytokines),w=4.8,h=4.2)
        for ax,c in zip(axs.flat,a.cytokines):
            measured(ax,c,show_arrow=False)
            cloud(ax,preds_xy[k,c],MODEL_COLORS[k],.40,12)
            if ('observed',c) in endpoints:draw_arrow(ax,start,endpoints['observed',c],OBS_COLOR,'--',2.2,'^')
            draw_arrow(ax,start,endpoints[k,c],MODEL_COLORS[k],'-',2.5,markers[k]);start_mark(ax)
            setup(ax,bb,labels);ax.set_title(c,fontweight='bold')
        title(fig,f'{a.holdout}: {model_name(k,200)}','Predicted latent cell distributions and centroid movement | common target-only UMAP')
        base_legend(fig,[pbs_handle,start_handle]+obs_handles+[marker_handle(MODEL_COLORS[k],'Predicted cells'),model_handles[k]],3)
        layout(fig,bottom=.15);save(fig,f'05_target_predictions_{k}')


def target_only_umap(z,obs,preds,centers,metrics,a):
    """Retrospective UMAP fitted only to measured target PBS + treated cells.

    All model predictions already exist. Predictions and latent centroids are
    transformed into this target-only map, never used to fit it. The observed
    target treatment is used for visualization geometry only, NOT prediction.
    """
    q=obs.cell_type.eq(a.holdout).to_numpy()
    ids=np.flatnonzero(q); target_z=z[ids]; oo=obs.iloc[ids].reset_index(drop=True)
    cc=oo.cytokine.to_numpy(); pbs=cc=='PBS'
    if len(target_z)<3: raise ValueError('Target-only UMAP requires at least three measured cells.')
    print(f'Fitting separate target-only UMAP on {len(target_z):,} measured {a.holdout} cells (retrospective display only).',flush=True)
    model,xy=umap_fit(target_z,a.n_neighbors,a.min_dist,a.state_metric,a.seed)
    transform=lambda v:safe_transform(model,v)
    kinds=list(dict.fromkeys(k for k,c in preds))
    keys=[('PBS','PBS')]; vectors=[centers[a.holdout,'PBS']]
    for c in a.cytokines:
        if (cc==c).any(): keys.append(('observed',c));vectors.append(target_z[cc==c].mean(0,dtype=np.float64))
        for k in kinds: keys.append((k,c));vectors.append(preds[k,c].mean(0,dtype=np.float64))
    endpoints=dict(zip(keys,transform(np.stack(vectors))))
    start=endpoints['PBS','PBS']
    folder=a.outdir/'figures'/'umap'/'target_only_umap';folder.mkdir(parents=True,exist_ok=True)
    oo['UMAP1']=xy[:,0];oo['UMAP2']=xy[:,1];oo['used_for_target_umap_fit']=True
    oo.to_csv(a.outdir/'target_only_umap_coordinates.csv',index=False)
    pd.DataFrame([dict(model=k,cytokine=c,UMAP1=v[0],UMAP2=v[1]) for (k,c),v in endpoints.items()]).to_csv(a.outdir/'target_only_umap_centroid_endpoints.csv',index=False)
    # Project predicted single-cell latent states onto this same measured-target
    # map; neither the predicted cells nor the centroids enter the UMAP fit.
    preds_xy={}
    for key,v in preds.items():
        preds_xy[key]=np.concatenate([transform(v[i:i+4096]) for i in range(0,len(v),4096)])
    pred_keys=list(preds_xy)
    np.savez_compressed(a.outdir/'target_only_umap_predictions.npz',
                        model=np.array([k for k,c in pred_keys]),
                        cytokine=np.array([c for k,c in pred_keys]),
                        xy=np.stack([preds_xy[k] for k in pred_keys]),pbs_xy=xy[pbs],
                        cell_id=oo.loc[pbs,'cell_id'].to_numpy(dtype=str))
    target_umap_overview(xy,cc,preds_xy,endpoints,kinds,folder,a)
    markers={'default':'D','hybrid':'h'}
    # A single large figure per cytokine. No other cell types or treatments shown.
    # Full-dimensional projected centroids, not mean UMAP coordinates.
    for ci,c in enumerate(a.cytokines,1):
        pp=xy[pbs]; tt=xy[cc==c]; actual=endpoints.get(('observed',c))
        pp_ep=np.stack([endpoints[k,c] for k in kinds]); cb=bounds(pp,tt,pp_ep,start,actual if actual is not None else start)
        fig,ax=plt.subplots(figsize=(11.5,7.5))
        cloud(ax,pp,PBS_COLOR,.48,19)
        cloud(ax,tt,OBS_COLOR,.46,20,'^')
        if actual is not None: draw_arrow(ax,start,actual,OBS_COLOR,'--',2.7,'^')
        for k in kinds: draw_arrow(ax,start,endpoints[k,c],MODEL_COLORS[k],'-',2.5,markers[k])
        ax.scatter(*start,s=115,marker='o',facecolors='white',edgecolors='#303C49',linewidths=2,zorder=10)
        setup(ax,cb,['Target UMAP 1','Target UMAP 2'])
        hs=[marker_handle(PBS_COLOR,'PBS cells'),Line2D([],[],ls='',marker='o',mfc='white',mec='#303C49',mew=1.7,label='PBS centroid',markersize=8)]
        if actual is not None: hs += [marker_handle(OBS_COLOR,'Observed treated cells','^'),marker_handle(OBS_COLOR,'Observed centroid','^','--')]
        hs += [marker_handle(MODEL_COLORS[k],model_name(k,200),markers[k],'-') for k in kinds]
        ax.legend(handles=hs,loc='upper left',bbox_to_anchor=(1.035,1),frameon=False,fontsize=10,labelspacing=1.15,title='Centroid movements',title_fontsize=11)
        title(fig,f'{a.holdout} | {c}', 'Target-only UMAP • PBS, measured treatment, shared scGen and hybrid' if actual is not None else 'Target-only UMAP • PBS, shared scGen and hybrid predictions')
        fig.text(.065,.025,'Retrospective target-only map; predictions fixed before visualization. Solid = predicted; dashed = measured.\nCentroids computed in 200D, then projected. UMAP distances and directions are illustrative.',fontsize=9,color='#506074',va='bottom')
        fig.subplots_adjust(left=.07,right=.73,top=.87,bottom=.14)
        fig.savefig(folder/f'{ci:02d}_{slug(c)}.png',dpi=a.dpi,bbox_inches='tight',facecolor='white');plt.close(fig)


def metric_figures(metrics,d,a):
    m=metrics[metrics.role=='heldout_target']; kinds=list(dict.fromkeys(m.model));
    fields=[('cosine_to_observed','Direction agreement (cosine)','RdBu_r',-1,1),('relative_centroid_error','Relative endpoint error','YlOrRd',0,None),('predicted_to_observed_norm_ratio','Shift magnitude ratio','viridis',0,None),('within_type_shift_rms','Within-type shift RMS','magma',0,None)]
    if m.cosine_to_observed.notna().sum()==0: fields=fields[-1:]
    fig,axs=plt.subplots(len(fields),1,figsize=(10,2.35*len(fields)+1.1),squeeze=False,layout='constrained')
    for ax,(f,lab,cmap,vmin,vmax) in zip(axs.flat,fields):
        v=m.pivot(index='model',columns='cytokine',values=f).reindex(index=kinds,columns=a.cytokines).to_numpy()
        high=vmax if vmax is not None else max(float(np.nanmax(v)) if np.isfinite(v).any() else 1,1e-8)
        im=ax.imshow(v,aspect='auto',cmap=cmap,vmin=vmin,vmax=high); ax.set_xticks(range(len(a.cytokines)),a.cytokines,fontsize=9)
        ax.set_yticks(range(len(kinds)),[model_name(k,d) for k in kinds],fontsize=9);ax.set_title(lab,fontweight='bold')
        for i in range(len(kinds)):
            for j in range(len(a.cytokines)):
                val=v[i,j]; rgba=im.cmap(im.norm(val)); light=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,f'{val:.2f}' if np.isfinite(val) else 'NA',ha='center',va='center',fontsize=9,color='white' if light<.47 else '#17202A')
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(f'Z{d}: full-dimensional target perturbation diagnostics',fontweight='bold',fontsize=16)
    save_fig(fig,'10_full_latent_prediction_metrics',a)


def run_space(scgen,ref,query,sources,run,d,a):
    b=copy.copy(a); b.outdir=a.outdir/f'Z{d}'; (b.outdir/'figures').mkdir(parents=True)
    z,obs,preds,source,digest=encode(scgen,ref,query,run,d,b,sources)
    centers,counts,meta,shifts,types,sources=reference_stats(z,obs,b)
    tab=list(plt.get_cmap('tab20').colors); tab=tab[::2]+tab[1::2]
    tc={t:tab[i%len(tab)] for i,t in enumerate(types)}; co={c:COND_COLORS.get(c,tab[i%len(tab)]) for i,c in enumerate(['PBS']+a.cytokines)}
    metrics=numeric_tables(z,obs,preds,source,centers,counts,meta,shifts,d,b)
    print('Z200: generating reference UMAP figures...',flush=True)
    projection_figures(z,obs,preds,source,centers,types,sources,metrics,d,b,tc,co)
    target_only_umap(z,obs,preds,centers,metrics,b)
    if not b.skip_direction_details:
        direction_figures(meta,shifts,tc,co,b)
    metric_figures(metrics,d,b)
    cfg={'latent_dim':d,'checkpoint':str(run),'vae_state_sha256':digest,'n_reference_cells':int(obs.status.eq('reference').sum()),'n_observed_target_cells':int(obs.status.eq('heldout_observed').sum()),'models':list(dict.fromkeys(k for k,c in preds)),'source_types':sources,'figure_format':'png','reference_umap_fit':'reference cells only','target_umap_fit':'measured target PBS and treated cells, visualization only','heatmap_order':'fixed alphabetical cell types and cytokine argument order; additional average-linkage cosine-clustered versions'}
    (b.outdir/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(f'Z{d} complete: {b.outdir}',flush=True)
    return metrics


GUIDE="""READING THE FIGURES
===================
Only shared scGen Z200 and the learned hybrid head are visualized. No model training, new gene selection or normalization.
All figures are PNG. All full-dimensional latent vectors and numeric tables remain.

Z200/figures/umap/ (one reference-fitted UMAP):
  01_cell_state: reference cellular states colored by cell type and treatment.
  02_states_by_treatment: observed reference distributions by condition.
  03_source_observed_arrows: measured source PBS-to-treated centroids, not predictions.
  04_target_all_model_arrows: six panels of predicted and measured target centroid movements.
  05_source_predictions_*: in-sample source model predictions versus measured
    source responses. Solid arrows = predicted; dashed = observed.

Z200/figures/umap/target_only_umap/ (separate target-focused UMAP):
  One large PNG per cytokine showing target PBS cells, measured treated target
  cells, the starting PBS centroid, the measured treated centroid, and both
  model-predicted centroids. No other cell types are displayed. A separate UMAP
  is fitted once on measured target PBS and all available measured target treated
  cells. Predictions and centroids are transformed into this map, not used for
  fitting. This is retrospective visualization, NOT independent model evaluation.
  All cytokines use the same target-only map, with plot limits zoomed per cytokine.
  If --no-observed-target is set, this map is fitted on target PBS only.
  overview/ contains corresponding target-only summaries, all on the same map
  and with identical limits across cytokines and models:
    01_cell_state: measured PBS versus treated and colors by treatment.
    02_states_by_treatment: measured distributions by condition.
    03_observed_target_arrows: measured PBS-to-treated centroid movements.
    04_target_all_model_arrows: shared scGen and hybrid versus observed.
    05_target_predictions_default: shared scGen predicted cell clouds and arrows.
    05_target_predictions_hybrid: hybrid predicted cell clouds and arrows.
  There is no target_by_cytokine directory on the global reference UMAP.

Z200/figures/ (full-dimensional shift diagnostics):
  04_UMAP_shift_directions: UMAP of nonzero measured source shift directions.
  05_latent_shift_directions: signed unit-vector components in original Z200 order.
  06_shift_cosine_similarity: pairwise measured shift cosine matrix, fixed order.
  07_cell_type_shift_similarity_per_cytokine: one matrix per cytokine; same
    alphabetical cell-type order on both axes across all panels.
  08_cytokine_shift_similarity_per_cell_type: one matrix per source type; same
    cytokine order (command-line order) on both axes across all panels.
  09_latent_shift_magnitudes: observed shift norms, same cell-type/cytokine order.
  10_full_latent_prediction_metrics: quantitative target centroid diagnostics.
  Figures 05, 06, 07 and 08 each have a second PNG with suffix _clustered.
  The original filenames retain the fixed-order versions. Clustering uses
  average linkage on cosine distance of the full-dimensional shift directions.
  05 clusters rows only, retaining the original 200 latent dimensions on columns;
  06 uses that same global permutation on both axes. Each clustered 07/08 panel
  independently clusters its own vectors with the same order on both axes.
  Fixed-order versions retain the same row/column order across panels.
  Both versions use the same color scales. Undefined directions (zero vectors)
  are NA and are appended after valid directions in clustered versions.
  Ordered matrices and display-order CSVs are saved for reproducibility.

HOW PREDICTIONS ARE GENERATED
Default shared scGen: add the exact saved cytokine-specific shift to every PBS
latent cell. Learned hybrid: apply its frozen weights to the PBS latent cells
and stored cytokine embeddings, using the original global branch plus nonlinear
correction with dropout disabled.
Predictions are fixed before target treated cells are encoded. The reference
UMAP is fitted on reference cells only; observed target cells do not affect it.
The separate target-only UMAP deliberately uses measured target treated cells
for retrospective visualization only; it does not change any predictions,
weights, or full-dimensional metrics. No donor split or reference-shift retrieval.

INTERPRETATION
UMAP is nonlinear. Arrows connect projections of 200D centroids, not trajectories.
Their length/angle does not preserve full latent geometry, and the projected
centroid need not equal the mean of plotted cells. Reference and target-only
UMAPs are separate maps; do not compare their coordinates directly. Within one
map, all model predictions use the same projection. Source panels are in-sample.
Predicted latent cells are pre-decoder states, not decoded/re-encoded expression.
Group-mean supervision does not establish accurate individual-cell responses.

Full-dimensional metrics: cosine direction agreement (1 is ideal); relative
centroid error (0 is ideal); magnitude ratio (1 is ideal); within-type shift RMS
(computational shift heterogeneity). These are not gene-expression Pearson/MSE.
All cells contribute to full-dimensional means and metrics. Optional caps affect
reference UMAP fitting or point display only. The target-only map uses all measured
target cells. CSV and NPZ outputs retain the full-dimensional data and map coordinates.
"""


def main():
    a=parse_args()
    # Preflight every input before performing expensive reference embedding.
    path=a.shared_z200/('shared__'+a.holdout.replace(' ','_').replace('/','-'))/'model.pt'
    if not path.is_file(): raise FileNotFoundError(path)
    read_saved_shifts(a.shared_z200,200,a)
    for k in a.heads:
        if not (a.heads_dir/f'head_{k}.pt').is_file(): raise FileNotFoundError(a.heads_dir/f'head_{k}.pt')
    try: import umap
    except ImportError as e: raise SystemExit('Install UMAP: python -m pip install umap-learn') from e
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12})
    scgen,ref,query,sources=load_input(a)
    a.outdir.mkdir(parents=True,exist_ok=True)
    (a.outdir/'FIGURE_GUIDE.txt').write_text(GUIDE)
    (a.outdir/'run_config.json').write_text(json.dumps(vars(a),indent=2,default=str)+'\n')
    metrics=run_space(scgen,ref,query,sources,a.shared_z200,200,a)
    gc.collect()
    metrics.to_csv(a.outdir/'all_latent_prediction_metrics.csv',index=False)
    print(f'Finished. All results: {a.outdir}',flush=True)

if __name__=='__main__': main()
