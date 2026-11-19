#building blocks for the attention VAE --> plain pytorch, no scvi / scgen imports

import torch
from torch import nn


def mlp_trunk(n_in: int, n_hidden: int = 800, n_layers: int = 2, dropout: float = 0.2) -> nn.Sequential:
    #scGen hidden stack: (Linear --> BatchNorm --> LeakyReLU --> Dropout) x n_layers
    layers = []
    for i in range(n_layers):
        layers += [
            nn.Linear(n_in if i == 0 else n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    #scGen encoder: x --> 800 --> 800 --> (mu, logvar)
    def __init__(self, n_input: int, n_latent: int = 100, n_hidden: int = 800, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.trunk = mlp_trunk(n_input, n_hidden, n_layers, dropout)
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x, return_attn: bool = False):
        h = self.trunk(x)
        return self.mu(h), self.logvar(h), None


class Decoder(nn.Module):
    #scGen decoder: z --> 800 --> 800 --> features (linear output, log-normalized expression or TF scores)
    def __init__(self, n_latent: int, n_output: int, n_hidden: int = 800, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.trunk = mlp_trunk(n_latent, n_hidden, n_layers, dropout)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, z):
        return self.out(self.trunk(z))


class GeneModuleTokenizer(nn.Module):
    #Model A tokenizer: 7000 genes --> K soft gene-module tokens
    #each gene g has embedding E_g and a soft assignment A[:, g] over K modules (softmax over modules)
    #token_k = sum_g A[k, g] * x_g * E_g --> attention cost is K^2 instead of G^2
    def __init__(self, n_genes: int, n_tokens: int = 64, d_model: int = 64):
        super().__init__()
        self.assign_logits = nn.Parameter(torch.randn(n_tokens, n_genes) * 0.01)
        self.gene_emb = nn.Parameter(torch.randn(n_genes, d_model) / d_model ** 0.5)
        self.token_bias = nn.Parameter(torch.zeros(n_tokens, d_model))
        self.n_tokens = n_tokens

    def assignment(self) -> torch.Tensor:
        #(K, G) gene --> module membership, columns sum to 1
        return torch.softmax(self.assign_logits, dim=0)

    def forward(self, x):
        #x (B, G) --> (B, K, G) weighted by assignment --> (B, K, d)
        a = self.assignment()
        return torch.einsum("kg,bg,gd->bkd", a, x, self.gene_emb) + self.token_bias


class FeatureTokenizer(nn.Module):
    #Model B tokenizer: one token per TF --> token_f = x_f * e_f + b_f
    #attention weights then point at named TFs
    def __init__(self, n_features: int, d_model: int = 64):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n_features, d_model) / d_model ** 0.5)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.n_tokens = n_features

    def forward(self, x):
        return x.unsqueeze(-1) * self.emb + self.bias


class AttnBlock(nn.Module):
    #pre-norm transformer block --> returns per-head attention weights when asked
    def __init__(self, d_model: int = 64, n_heads: int = 4, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, return_attn: bool = False):
        q = self.norm1(h)
        out, w = self.attn(q, q, q, need_weights=return_attn, average_attn_weights=False)
        h = h + self.drop(out)
        h = h + self.drop(self.ff(self.norm2(h)))
        return h, w  #w: (B, heads, T, T) or None


class TokenAttentionEncoder(nn.Module):
    #x --> tokens --> [CLS] + tokens --> attention blocks --> pool [CLS ; mean] --> (mu, logvar)
    #hybrid=True --> pooled attention features are added to the scGen MLP trunk before the heads
    def __init__(
        self,
        n_input: int,
        n_latent: int = 100,
        space: str = "genes",
        n_tokens: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        attn_dropout: float = 0.1,
        hybrid: bool = False,
        n_hidden: int = 800,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if space == "genes":
            self.tokenizer = GeneModuleTokenizer(n_input, n_tokens, d_model)
        elif space == "tf":
            self.tokenizer = FeatureTokenizer(n_input, d_model)
        else:
            raise ValueError(f"space must be 'genes' or 'tf', got {space}")
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([AttnBlock(d_model, n_heads, dropout=attn_dropout) for _ in range(n_attn_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.hybrid = hybrid
        if hybrid:
            self.trunk = mlp_trunk(n_input, n_hidden, n_layers, dropout)
            self.proj = nn.Linear(2 * d_model, n_hidden)
            n_head_in = n_hidden
        else:
            self.proj = nn.Sequential(nn.Linear(2 * d_model, n_hidden), nn.LeakyReLU(), nn.Dropout(dropout))
            n_head_in = n_hidden
        self.mu = nn.Linear(n_head_in, n_latent)
        self.logvar = nn.Linear(n_head_in, n_latent)

    def forward(self, x, return_attn: bool = False):
        tok = self.tokenizer(x)  #(B, T, d)
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        attns = []
        for blk in self.blocks:
            h, w = blk(h, return_attn)
            if return_attn:
                attns.append(w)
        h = self.norm(h)
        pooled = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        feat = self.proj(pooled)
        if self.hybrid:
            feat = self.trunk(x) + feat
        attn = torch.stack(attns, dim=1) if return_attn else None  #(B, layers, heads, T+1, T+1)
        return self.mu(feat), self.logvar(feat), attn
