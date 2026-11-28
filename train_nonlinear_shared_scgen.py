#!/usr/bin/env python3
"""Train four perturbation heads on a frozen shared scGen Z200 VAE.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd
import torch
from torch import nn

CYTOKINES = ['IFN-beta', 'IFN-gamma', 'IL-1-beta', 'IL-15', 'IL-32-beta', 'IL-4']
RARE = ['pDC', 'HSPC', 'ILC', 'Plasmablast']
METRICS = ['r_all', 'r_top100', 'mse']
TOL = {'r_all': 0.0000501, 'r_top100': 0.0000501, 'mse': 0.000000501}
LABELS = {
    'original_naive': 'Original naive',
    'original_scgen_Z100': 'Original scGen Z100',
    'shared_scgen_Z100': 'Shared scGen Z100',
    'shared_scgen_Z200': 'Shared scGen Z200 (default)',
    'learned_global': 'Learned global shift',
    'learned_affine': 'Learned affine shift',
    'learned_nonlinear': 'Learned nonlinear shift',
    'learned_hybrid': 'Learned hybrid shift',
}


def dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor(x, device):
    # Fresh ordinary tensors, not inference-mode tensors; autograd may save them.
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)


class PerturbationHead(nn.Module):
    def __init__(self, latent_dim, n_cytokines, embedding_dim=32, hidden=256,
                 dropout=0.1, kind='nonlinear'):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(n_cytokines, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0., std=.1)
        if kind in ('nonlinear', 'hybrid'):
            self.net = nn.Sequential(nn.Linear(latent_dim + embedding_dim, hidden),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, latent_dim))
        elif kind == 'affine':
            self.net = nn.Sequential(nn.Linear(latent_dim + embedding_dim, latent_dim))
        elif kind == 'global':
            self.net = nn.Sequential(nn.Linear(embedding_dim, latent_dim))
        else:
            raise ValueError(kind)
        # Start as the unperturbed frozen VAE; learn residual shifts from scratch.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if kind == 'hybrid':
            # Explicit global cytokine branch, with the SAME trainable embedding
            # as the nonlinear correction. Both branches are optimized jointly.
            self.global_net = nn.Linear(embedding_dim, latent_dim)
            nn.init.zeros_(self.global_net.weight)
            nn.init.zeros_(self.global_net.bias)

    def shift_components(self, z, cytokine_idx):
        """Return the two additive hybrid components; each is N x latent_dim.

        The global term is constant across cells at inference; the correction
        is unrestricted (not constrained to have zero mean).
        """
        if self.kind != 'hybrid':
            raise ValueError('shift_components is defined for the hybrid head only')
        ids = torch.full((len(z),), int(cytokine_idx), dtype=torch.long, device=z.device)
        e = self.embedding(ids)
        return self.global_net(e), self.net(torch.cat([z, e], dim=1))

    def shift(self, z, cytokine_idx):
        if self.kind == 'hybrid':
            global_shift, correction = self.shift_components(z, cytokine_idx)
            return global_shift + correction
        ids = torch.full((len(z),), int(cytokine_idx), dtype=torch.long, device=z.device)
        e = self.embedding(ids)
        return self.net(e if self.kind == 'global' else torch.cat([z, e], dim=1))

    def forward(self, z, cytokine_idx):
        return z + self.shift(z, cytokine_idx)


def model_digest(module):
    """Check all VAE parameters AND buffers, including batch-normalization state."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decode(module, z):
    # IMPORTANT: no no_grad here during training. Decoder parameters are frozen,
    # but gradients must flow through its operations to the perturbation head.
    return module.generative(z)['px']


def group_predict(module, head, z, ci, batch_size, delta=None, return_cells=False):
    """Predict a population with dropout off; decode EACH cell then average."""
    module.eval()
    if head is not None:
        head.eval()
    psum = zsum = None
    expr_parts, shift_parts = [], []
    with torch.no_grad():
        for block in z.split(batch_size):
            if head is not None:
                pert = head(block, ci)
            elif delta is not None:
                pert = block + delta
            else:
                pert = block
            pred = decode(module, pert)
            s1, s2 = pred.sum(0), pert.sum(0)
            psum = s1 if psum is None else psum + s1
            zsum = s2 if zsum is None else zsum + s2
            if return_cells:
                expr_parts.append(pred.cpu().numpy())
                shift_parts.append((pert - block).cpu().numpy())
    if return_cells:
        return psum / len(z), zsum / len(z), np.concatenate(expr_parts), np.concatenate(shift_parts)
    return psum / len(z), zsum / len(z)


def group_loss(pred_mean, latent_mean, group, latent_weight):
    # Both predicted and measured responses use the identical real PBS mean.
    # It cancels in the difference, but the explicit form shows the objective.
    pred_delta = pred_mean - group['pbs_mean_x']
    real_delta = group['treated_mean_x'] - group['pbs_mean_x']
    gene = (pred_delta - real_delta).square().mean()
    latent = (latent_mean - group['treated_mean_z']).square().mean()
    return gene + latent_weight * latent, gene, latent


def train_epoch(module, head, groups, optimizer, rng, args):
    module.eval()  # do not update decoder/encoder batchnorm or dropout
    head.train()
    losses = []
    for j in rng.permutation(len(groups)):
        group = groups[j]
        z = group['pbs_z']
        # One balanced step per source cell type / cytokine group. No artificial
        # pairing of individual untreated and treated cells.
        if len(z) > args.batch_size:
            idx = torch.as_tensor(rng.choice(len(z), args.batch_size, replace=False), device=z.device)
            z = z[idx]
        optimizer.zero_grad(set_to_none=True)
        zp = head(z, group['cytokine_idx'])
        xp = decode(module, zp)
        loss, gene, latent = group_loss(xp.mean(0), zp.mean(0), group, args.latent_loss_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite head loss; consider a smaller learning rate.')
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        losses.append([loss.item(), gene.item(), latent.item()])
    return np.mean(losses, axis=0)


def fit_head(module, groups, latent_dim, ncyt, sources, kind, args):
    """Fit once on all source groups for exactly args.epochs epochs."""
    seed_all(args.seed)
    head = PerturbationHead(latent_dim, ncyt, args.embedding_dim, args.hidden,
                            args.dropout, kind).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 1)
    history = []
    print(f'{kind}: training once for {args.epochs} epochs on all {len(groups)} source groups.')
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(module, head, groups, optimizer, rng, args)
        history.append(dict(head=kind, phase='training', epoch=epoch,
                            train_loss=tr[0], train_gene_mse=tr[1],
                            train_latent_mse=tr[2]))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'{kind:9s} epoch {epoch:3d}/{args.epochs} | train gene MSE={tr[1]:.6f}', flush=True)
    head.eval()
    return head, history, dict(head=kind, training_mode='fixed_epochs',
                               final_training_cell_types=list(sources),
                               training_epochs=args.epochs,
                               trainable_parameters=sum(p.numel() for p in head.parameters()))


def score(pred, base, real_delta):
    # Same arithmetic and ranking as scgen_shared.py and final ablation.
    d = np.asarray(pred) - np.asarray(base)
    top = np.argsort(-np.abs(real_delta))[:100]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = dict(r_all=float(np.corrcoef(d, real_delta)[0, 1]),
                   r_top100=float(np.corrcoef(d[top], real_delta[top])[0, 1]),
                   mse=float(np.mean((d - real_delta)**2)))
    return out


def assert_match(a, b, label):
    for m in METRICS:
        if not np.isclose(float(a[m]), float(b[m]), atol=TOL[m], rtol=0):
            raise RuntimeError(f'Benchmark mismatch {label}/{m}: {a[m]} vs {b[m]}. Check data/checkpoint.')


def read_saved_scores(path, args, historical=False):
    d = pd.read_csv(path)
    if historical:
        d = d[d['run'] == args.historical_run]
    d = d[d.holdout.eq(args.holdout) & d.cytokine.isin(args.cytokines) &
          d.model.isin(['naive', 'scGen', 'scGen_vs_real_ctrl'])]
    expected = {(c, m) for c in args.cytokines for m in ['naive', 'scGen', 'scGen_vs_real_ctrl']}
    if set(zip(d.cytokine, d.model)) != expected or d.duplicated(['cytokine', 'model']).any():
        raise ValueError(f'Incomplete/duplicate saved scores: {path}')
    return d.set_index(['cytokine', 'model'])


def write_plots(scores, summary, history, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    folder = args.outdir / 'figures'; folder.mkdir(exist_ok=True)
    def save(fig, name):
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(folder / f'{name}.{ext}', dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
    metrics = [('r_all', 'Pearson r | all genes'), ('r_top100', 'Pearson r | top 100 genes'), ('mse', 'MSE')]
    for ref in ['real_PBS', 'reconstructed_PBS']:
        sub = scores[scores.reference == ref]
        order = [k for k in LABELS if k in set(sub.family)]
        if not order:
            continue
        palette = {'original_naive': '#929BA5', 'original_scgen_Z100': '#64778D',
                   'shared_scgen_Z100': '#64778D', 'shared_scgen_Z200': '#64778D',
                   'learned_global': '#4477AA', 'learned_affine': '#CC8844',
                   'learned_nonlinear': '#007F71', 'learned_hybrid': '#884E9A'}
        colors = [palette[k] for k in order]
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4, .49 * len(order) + 1.6)), layout='constrained')
        for j, (ax, (m, title)) in enumerate(zip(axes, metrics)):
            for i, k in enumerate(order):
                vals = sub[sub.family.eq(k)][m].to_numpy()
                ax.scatter(vals, i + np.linspace(-.12,.12,len(vals)), color=colors[i], alpha=.45, s=25, zorder=3)
                ax.scatter(vals.mean(), i, marker='D', color=colors[i], edgecolors='white', s=88, zorder=4)
            ax.set_yticks(np.arange(len(order)), [LABELS[k] for k in order] if j == 0 else [''] * len(order))
            ax.set_ylim(len(order)-.55,-.55); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
            ax.set_title(title, fontweight='bold'); ax.set_xlabel('Lower is better' if m=='mse' else 'Higher is better')
        fig.suptitle(f'Cytokine-conditioned perturbation prediction | {args.holdout}\n{ref.replace("_", " ")} | dots: cytokines; diamonds: mean', fontsize=15, fontweight='bold')
        save(fig, f'01_comparison_{ref}')
        fig, axes = plt.subplots(1, 3, figsize=(15.5,max(4.5,.46*len(order)+2)), layout='constrained')
        for j, (ax, (m, title)) in enumerate(zip(axes, metrics)):
            a = sub.pivot(index='family',columns='cytokine',values=m).reindex(index=order,columns=args.cytokines).to_numpy()
            im = ax.imshow(a, aspect='auto', cmap='YlGnBu_r' if m=='mse' else 'YlGnBu')
            for i in range(a.shape[0]):
                for l in range(a.shape[1]):
                    v=a[i,l]; rgba=im.cmap(im.norm(v)); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(l,i,f'{v:.4f}' if m=='mse' else f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if lum<.43 else '#14202C')
            for l in range(a.shape[1]):
                i=int(np.argmin(a[:,l]) if m=='mse' else np.argmax(a[:,l]))
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((l-.5,i-.5),1,1,fill=False,edgecolor='#E7A72B',linewidth=1.8))
            ax.set_xticks(range(len(args.cytokines)),args.cytokines,rotation=45,ha='right')
            ax.set_yticks(range(len(order)),[LABELS[k] for k in order] if j==0 else ['']*len(order))
            ax.set_title(title,fontweight='bold'); fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.055,pad=.16)
        fig.suptitle(f'Per-cytokine prediction scores | {ref.replace("_", " ")}\nOutlined cells: best score',fontsize=15,fontweight='bold')
        save(fig,f'02_per_cytokine_{ref}')
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    for kind, d in history.groupby('head', sort=False):
        ax.plot(d.epoch, d.train_gene_mse, label=LABELS['learned_'+kind],
                color=palette['learned_'+kind])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Group-mean gene MSE')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    ax.set_title('Perturbation-head training | frozen VAE', fontweight='bold')
    save(fig,'03_training_curves')


def load_data(args):
    import anndata as ad
    import scgen
    from scgen._scgenvae import SCGENVAE
    # Identical compatibility patch to scgen_shared.py and the successful ablation.
    orig = SCGENVAE.inference
    def inference_fixed(self, x):
        o = orig(self, x)
        if 'qz_m' in o: o.setdefault('qzm', o['qz_m'])
        if 'qz_v' in o: o.setdefault('qzv', o['qz_v'])
        return o
    SCGENVAE.inference = inference_fixed
    tag = 'shared__' + args.holdout.replace(' ', '_').replace('/', '-')
    checkpoint = args.run / tag
    if not (checkpoint / 'model.pt').is_file():
        raise FileNotFoundError(f'Missing frozen model: {checkpoint}')
    data = ad.read_h5ad(args.data)
    data = data[data.obs.cytokine.isin(['PBS']+args.cytokines)].copy()
    if not args.keep_rare:
        data = data[~data.obs.cell_type.isin(RARE)].copy()
    data.obs['condition'] = data.obs.cytokine.astype(str)
    data.obs['cell_type'] = data.obs.cell_type.astype(str)
    if 'counts' in data.layers: del data.layers['counts']
    if not data.var_names.is_unique or not data.obs_names.is_unique:
        raise ValueError('Gene/cell names must be unique.')
    ct, cc = data.obs.cell_type.to_numpy(), data.obs.condition.to_numpy()
    if args.holdout not in set(ct): raise ValueError('Missing held-out type')
    sources = sorted(set(ct)-{args.holdout})
    if not sources: raise ValueError('No source cell types available')
    for t in sources+[args.holdout]:
        for c in ['PBS']+args.cytokines:
            if not np.any((ct==t)&(cc==c)): raise ValueError(f'Missing group {t}/{c}')
    held = (ct==args.holdout)&(cc!='PBS')
    train = data[~held].copy()
    train.X = dense(train.X).astype(np.float32,copy=False)
    print(f'Loading frozen VAE: {checkpoint}\nVAE reference: {train.n_obs:,} cells, {data.n_vars:,} genes; target treated excluded: {held.sum():,}')
    model = scgen.SCGEN.load(str(checkpoint), adata=train, accelerator='cpu', device='auto')
    model.to_device(args.device); model.module.eval()
    for p in model.module.parameters(): p.requires_grad_(False)
    with torch.inference_mode():
        z = np.asarray(model.get_latent_representation(give_mean=True,batch_size=args.eval_batch_size), dtype=np.float32)
    if z.shape != (train.n_obs,200) or not np.isfinite(z).all():
        raise ValueError(f'Expected frozen Z200 checkpoint, received {z.shape}')
    tc, c0 = train.obs.cell_type.to_numpy(), train.obs.condition.to_numpy()
    assert not np.any((tc==args.holdout)&(c0!='PBS'))
    return data, ct, cc, train, tc, c0, z, sources, model, checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=Path('data/hcd_subset.h5ad'))
    p.add_argument('--run',type=Path,default=Path('data/shared_CD14_Mono_Z200'))
    p.add_argument('--shared-z100',type=Path,default=Path('data/shared_CD14_Mono'))
    p.add_argument('--historical-csv',type=Path,default=Path('analysis/results_L2_vs_L4_H800_Z100.csv'))
    p.add_argument('--historical-run',default='scgen_runs_L2_H800_Z100')
    p.add_argument('--holdout',default='CD14 Mono')
    p.add_argument('--cytokines',nargs='+',default=CYTOKINES)
    p.add_argument('--keep-rare',action='store_true')
    p.add_argument('--heads',nargs='+',choices=['nonlinear','affine','global','hybrid'],default=['global','affine','nonlinear','hybrid'],help='Train all four heads by default, or specify a subset.')
    p.add_argument('--embedding-dim',type=int,default=32)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=100,help='Fixed training epochs per head on all source cell types.')
    p.add_argument('--batch-size',type=int,default=128,help='PBS cells sampled per source group per optimizer step.')
    p.add_argument('--eval-batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--grad-clip',type=float,default=5.)
    p.add_argument('--latent-loss-weight',type=float,default=0.,help='Optional auxiliary treated latent-centroid MSE; default gene-delta loss only.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--outdir',type=Path,default=Path('data/nonlinear_shared_Z200_E32_fixed100'))
    p.add_argument('--save-predictions',action='store_true',help='Also save per-cell predicted expression as compressed NPZ.')
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--log-every',type=int,default=5)
    args=p.parse_args()
    if 'PBS' in args.cytokines or len(set(args.cytokines))!=len(args.cytokines) or len(set(args.heads))!=len(args.heads): p.error('Use unique treated cytokines and heads.')
    if min(args.embedding_dim,args.hidden,args.epochs,args.batch_size,args.eval_batch_size,args.log_every,args.dpi)<1 or args.seed<0 or not 0<=args.dropout<1 or args.lr<=0 or args.grad_clip<=0 or min(args.weight_decay,args.latent_loss_weight)<0: p.error('Invalid hyperparameters.')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; use --device cpu.')
    if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())): p.error('Use a new or empty --outdir.')
    for path in [args.run,args.shared_z100]:
        if args.outdir.resolve()==path.resolve() or path.resolve() in args.outdir.resolve().parents: p.error('Keep output separate from saved model runs.')
    # Require the three saved baseline tables before launching training.
    benchmark_specs = [
        ('original_scgen_Z100', args.historical_csv, True),
        ('shared_scgen_Z100', args.shared_z100/'results.csv', False),
        ('shared_scgen_Z200', args.run/'results.csv', False),
    ]
    tables = {}
    for label, path, historical in benchmark_specs:
        if not path.is_file():
            raise FileNotFoundError(f'Required saved benchmark missing: {path}')
        tables[label] = read_saved_scores(path, args, historical)
        print(f'Loaded saved benchmark: {label} ({path})')
    seed_all(args.seed)
    data, ct, cc, train, tc, c0, z, sources, model, checkpoint = load_data(args)
    module=model.module; before=model_digest(module)
    args.outdir.mkdir(parents=True,exist_ok=True)
    # Materialize source group means and source PBS embeddings ONLY. No target
    # treated expression or latent representations are accessed in this stage.
    pbs={}; groups=[]; gene_pbs=[]
    def mx(t,c): return np.asarray(dense(data.X[(ct==t)&(cc==c)]).mean(0),dtype=np.float32).reshape(-1)
    for t in sources:
        zp=z[(tc==t)&(c0=='PBS')]; xp=mx(t,'PBS')
        pbs[t]=tensor(zp,args.device); gene_pbs.append(xp)
        for ci,c in enumerate(args.cytokines):
            zm=z[(tc==t)&(c0==c)].mean(0)
            groups.append(dict(cell_type=t,cytokine=c,cytokine_idx=ci,pbs_z=pbs[t],
                               pbs_mean_x=tensor(xp,args.device),treated_mean_x=tensor(mx(t,c),args.device),
                               treated_mean_z=tensor(zm,args.device)))
    assert {g['cell_type'] for g in groups}==set(sources) and args.holdout not in sources
    print(f'Head training: {len(sources)} source cell types x {len(args.cytokines)} cytokines = {len(groups)} unpaired groups.')
    heads={}; histories=[]; designs=[]
    genes=np.asarray(data.var_names.astype(str),dtype=str)
    for kind in args.heads:
        h, hist, design=fit_head(module,groups,200,len(args.cytokines),sources,kind,args)
        heads[kind]=h; histories.extend(hist); designs.append(design)
        checkpoint_head=dict(state_dict={k:v.detach().cpu() for k,v in h.state_dict().items()},
            kind=kind,latent_dim=200,embedding_dim=args.embedding_dim,hidden=args.hidden,dropout=args.dropout,
            cytokines=args.cytokines,genes=genes.tolist(),frozen_vae=str(checkpoint.resolve()),vae_state_sha256=before,
            source_types=sources,holdout=args.holdout,training_design=design)
        torch.save(checkpoint_head,args.outdir/f'head_{kind}.pt')
        emb=pd.DataFrame(h.embedding.weight.detach().cpu().numpy(),columns=[f'E{i+1}' for i in range(args.embedding_dim)])
        emb.insert(0,'cytokine',args.cytokines); emb.to_csv(args.outdir/f'cytokine_embeddings_{kind}.csv',index=False)
    history=pd.DataFrame(histories); history.to_csv(args.outdir/'training_history.csv',index=False)
    after=model_digest(module)
    if before!=after or any(p.requires_grad for p in module.parameters()) or any(p.grad is not None for p in module.parameters()):
        raise RuntimeError('Frozen VAE state changed or acquired gradients.')
    print('Frozen VAE parameters and buffers verified unchanged. Head training complete.')
    # Only now evaluate target predictions. Hyperparameters and epochs are fixed.
    ctrl=mx(args.holdout,'PBS'); target_z=tensor(z[(tc==args.holdout)&(c0=='PBS')],args.device)
    recon=group_predict(module,None,target_z,0,args.eval_batch_size)[0].cpu().numpy()
    gene_pbs=np.stack(gene_pbs)
    scores=[]; checks=[]; target_means=[]; shift_summary=[]; hybrid_components=[]; hybrid_globals=[]
    for ci,c in enumerate(args.cytokines):
        observed=mx(args.holdout,c); real_delta=observed-ctrl
        gdelta=np.stack([mx(t,c) for t in sources])-gene_pbs
        def add(label,reference,pred,base,origin='computed'):
            m=score(pred,base,real_delta)
            scores.append(dict(cytokine=c,holdout=args.holdout,family=label,reference=reference,origin=origin,**m))
            return m
        naive=add('original_naive','real_PBS',ctrl+gdelta.mean(0),ctrl)
        check=dict(cytokine=c,frozen_vae_unchanged=(before==after),target_treated_used_for_head_training=False)
        for label,table in tables.items():
            assert_match(naive,table.loc[(c,'naive')],f'{label}/{c}/naive')
            check[label+'_naive_matches']=True
            for method,ref in [('scGen_vs_real_ctrl','real_PBS'),('scGen','reconstructed_PBS')]:
                m=table.loc[(c,method),METRICS].astype(float).to_dict()
                scores.append(dict(cytokine=c,holdout=args.holdout,family=label,reference=ref,origin='historical' if label=='original_scgen_Z100' else 'saved',**m))
        # Confirm reconstruction against checkpoint's saved output where available.
        recon_file=args.run/f'{c}__{args.holdout.replace(" ","_").replace("/","-")}_recon.h5ad'
        if recon_file.is_file():
            import anndata as ad
            ar=ad.read_h5ad(recon_file)
            if not ar.var_names.equals(data.var_names) or ar.n_obs!=len(target_z): raise RuntimeError(f'Saved reconstruction shape/gene order mismatch: {recon_file}')
            check['saved_recon_mse']=float(np.mean((dense(ar.X).mean(0)-recon)**2)); del ar
            if check['saved_recon_mse']>1e-7: raise RuntimeError('Frozen checkpoint does not reproduce saved PBS reconstruction.')
        for kind,h in heads.items():
            if args.save_predictions:
                mean_pred,mean_z,cell_pred,shifts=group_predict(module,h,target_z,ci,args.eval_batch_size,return_cells=True)
                np.savez_compressed(args.outdir/f'prediction_{kind}_{c}.npz',predicted_expression=cell_pred,
                    latent_shifts=shifts,genes=genes,cell_ids=np.asarray(train.obs_names[(tc==args.holdout)&(c0=='PBS')],dtype=str))
            else:
                mean_pred,mean_z=group_predict(module,h,target_z,ci,args.eval_batch_size)
                with torch.no_grad(): shifts=np.concatenate([h.shift(block,ci).cpu().numpy() for block in target_z.split(args.eval_batch_size)])
            pred=mean_pred.cpu().numpy(); label='learned_'+kind
            for ref,base in [('real_PBS',ctrl),('reconstructed_PBS',recon)]: add(label,ref,pred,base)
            target_means.append(dict(head=kind,cytokine=c,pred=pred,shift=shifts))
            if kind == 'hybrid':
                # Global branch is exactly shared by every cell for a cytokine.
                with torch.no_grad():
                    g0, _ = h.shift_components(target_z[:1], ci)
                gv = g0[0].cpu().numpy()
                hybrid_globals.append(dict(cytokine=c,global_shift_norm=float(np.linalg.norm(gv)),
                                           **{f'dZ{i+1}':v for i,v in enumerate(gv)}))
            for name,zz in list(pbs.items())+[(args.holdout,target_z)]:
                with torch.no_grad(): ss=torch.cat([h.shift(b,ci) for b in zz.split(args.eval_batch_size)])
                center=ss.mean(0); rms=(ss-center).square().sum(1).mean().sqrt()
                shift_summary.append(dict(head=kind,cytokine=c,cell_type=name,role='heldout_PBS' if name==args.holdout else 'source',
                     n_cells=len(zz),mean_shift_norm=center.norm().item(),within_type_shift_rms=rms.item(),**{f'dZ{i+1}':v for i,v in enumerate(center.cpu().numpy())}))
                if kind == 'hybrid':
                    with torch.no_grad():
                        components = [h.shift_components(b, ci) for b in zz.split(args.eval_batch_size)]
                        gg = torch.cat([a for a,b in components])
                        rr = torch.cat([b for a,b in components])
                        if not torch.allclose(gg, gg[:1].expand_as(gg), rtol=1e-5, atol=1e-6):
                            raise RuntimeError('Hybrid global component varies across cells.')
                        if not torch.allclose(gg+rr, ss, rtol=1e-5, atol=1e-6):
                            raise RuntimeError('Hybrid component sum does not reproduce total shift.')
                        for label_component, values in [('global',gg),('nonlinear_correction',rr),('combined',ss)]:
                            mean_component = values.mean(0)
                            component_rms = (values-mean_component).square().sum(1).mean().sqrt()
                            hybrid_components.append(dict(cytokine=c,cell_type=name,
                                role='heldout_PBS' if name==args.holdout else 'source',
                                component=label_component,n_cells=len(zz),
                                mean_shift_norm=mean_component.norm().item(),within_type_shift_rms=component_rms.item(),
                                **{f'dZ{i+1}':v for i,v in enumerate(mean_component.cpu().numpy())}))
        checks.append(check)
        print(pd.DataFrame([r for r in scores if r['cytokine']==c and r['reference']=='real_PBS'])[['family']+METRICS].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    scores=pd.DataFrame(scores)
    expected = {(c, f, ref) for c in args.cytokines
                for f in ['original_naive', *tables, *('learned_'+k for k in args.heads)]
                for ref in (['real_PBS'] if f == 'original_naive' else ['real_PBS','reconstructed_PBS'])}
    if set(zip(scores.cytokine, scores.family, scores.reference)) != expected:
        raise RuntimeError('Incomplete or unexpected model comparison rows.')
    if scores.duplicated(['cytokine','family','reference']).any(): raise RuntimeError('Duplicate scores')
    if not np.isfinite(scores[METRICS].to_numpy()).all(): raise RuntimeError('Nonfinite metric (constant delta or bad prediction); inspect training.')
    summary=scores.groupby(['holdout','family','reference'],sort=False)[METRICS].agg(['mean','std','count'])
    summary.columns=['_'.join(x) for x in summary.columns]; summary=summary.reset_index()
    scores.to_csv(args.outdir/'results.csv',index=False,float_format='%.10g')
    summary.to_csv(args.outdir/'summary.csv',index=False,float_format='%.10g')
    pd.DataFrame(checks).to_csv(args.outdir/'validation.csv',index=False)
    pd.DataFrame(shift_summary).to_csv(args.outdir/'learned_shift_summary.csv',index=False,float_format='%.8g')
    if hybrid_components:
        pd.DataFrame(hybrid_components).to_csv(args.outdir/'hybrid_shift_components.csv',index=False,float_format='%.8g')
        pd.DataFrame(hybrid_globals).to_csv(args.outdir/'hybrid_global_shifts.csv',index=False,float_format='%.8g')
    np.savez_compressed(args.outdir/'target_prediction_means.npz',genes=genes,heads=np.array([r['head'] for r in target_means]),
        cytokines=np.array([r['cytokine'] for r in target_means]),predicted_expression=np.stack([r['pred'] for r in target_means]),
        target_PBS_expression=ctrl,reconstructed_PBS_expression=recon,observed_treated_expression=np.stack([mx(args.holdout,r['cytokine']) for r in target_means]),
        learned_cell_shifts=np.stack([r['shift'] for r in target_means]),cell_ids=np.asarray(train.obs_names[(tc==args.holdout)&(c0=='PBS')],dtype=str))
    contrast=[]
    for kind in args.heads:
        baselines = ['original_naive','original_scgen_Z100','shared_scgen_Z100','shared_scgen_Z200']
        if kind == 'hybrid':
            baselines += ['learned_'+k for k in args.heads if k != kind]
        for baseline in baselines:
            a=scores[scores.family.eq('learned_'+kind)]; b=scores[scores.family.eq(baseline)]
            pair=a.merge(b,on=['cytokine','holdout','reference'],suffixes=('_learned','_baseline'),validate='one_to_one')
            for m in METRICS:
                for _,r in pair.iterrows():
                    gain=(r[m+'_baseline']-r[m+'_learned']) if m=='mse' else (r[m+'_learned']-r[m+'_baseline'])
                    contrast.append(dict(head=kind,baseline=baseline,cytokine=r.cytokine,reference=r.reference,metric=m,gain=gain,win=gain>0))
    contrast=pd.DataFrame(contrast); contrast.to_csv(args.outdir/'paired_comparisons.csv',index=False)
    contrast.groupby(['head','baseline','reference','metric'],sort=False).agg(mean_gain=('gain','mean'),wins=('win','sum'),n=('win','size')).reset_index().to_csv(args.outdir/'paired_comparison_summary.csv',index=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(checkpoint=str(checkpoint),vae_state_sha256=before,vae_unchanged=True,n_genes=data.n_vars,n_vae_reference_cells=train.n_obs,
                  n_source_types=len(sources),n_source_groups=len(groups),training_design=designs,heldout_treated_used_only_for_final_evaluation=True)
    if 'hybrid' in args.heads:
        config['hybrid_design'] = dict(formula='z + global_net(e_c) + nonlinear_net(concat(z,e_c))',
            shared_embedding=True, joint_training=True, warm_start=False, zero_initialized_outputs=True,
            correction_zero_mean_constraint=False)
    for pkg in ['torch','scgen','scvi-tools','anndata','numpy','pandas','matplotlib']:
        try: config.setdefault('versions',{})[pkg]=version(pkg)
        except PackageNotFoundError: pass
    (args.outdir/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.outdir/'METHODS.txt').write_text('''Frozen shared Z200 scGen: only the residual head and cytokine embedding are trained. No target treated cells, donor split, cell-type embedding or KNN enter head training. The target PBS cells were seen by the original VAE but are not head-training examples. Source groups contribute equally, one optimizer step per cell type/cytokine per epoch. Unpaired measured treated means are targets. Gene loss is MSE of predicted mean expression minus real PBS versus observed treated mean minus real PBS (equivalent to MSE of treated means). A random PBS subset approximates the full group mean during training. Optional latent centroid loss is controlled by --latent-loss-weight (default zero). Each head is initialized once and trained for the specified fixed number of epochs on all source groups. There is no source-validation split, epoch selection or refit. Group means do not establish accurate single-cell response distributions. Only the original naive and three saved scGen baselines are retained. CD14 Mono has been used for model development, so the comparison remains exploratory; independent holdouts and seeds are needed. Naive is shown only with real PBS.\n''')
    if 'hybrid' in args.heads:
        with (args.outdir/'METHODS.txt').open('a') as f:
            f.write('Hybrid head: a single learned cytokine embedding is shared between a linear global branch and a two-hidden-layer GELU correction branch. The total shift is their sum. Both branches start with zero output and train jointly from scratch under the same loss, fixed-epoch training procedure as the other heads. The correction is not constrained to have zero mean, so global and correction components are descriptive, not a uniquely identifiable decomposition of biology. Component means, magnitudes and within-type RMS are saved separately. No additional pretraining, reference-shift retrieval or target-type information is used by the hybrid head.\n')
    write_plots(scores,summary,history,args)
    print('\nFINAL MEANS (real PBS)\n',summary[summary.reference.eq('real_PBS')][['family']+[m+'_mean' for m in METRICS]].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print(f'All outputs: {args.outdir}')


if __name__=='__main__':
    main()
