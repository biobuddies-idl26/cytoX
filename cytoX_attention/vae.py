#scGen-style VAE with a swappable encoder --> mlp (scGen), attention, or hybrid
from typing import Optional

import torch
from torch import nn

from .modules import Decoder, MLPEncoder, TokenAttentionEncoder


class AttentionVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int = 100,
        encoder_type: str = "hybrid",
        space: str = "genes",
        n_hidden: int = 800,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        n_layers: Optional[int] = None,
        dropout: float = 0.2,
        n_tokens: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        attn_dropout: float = 0.1,
        ff_mult: int = 2,
        kl_weight: float = 5e-5,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        if n_layers is not None:
            n_enc_layers = n_dec_layers = n_layers  #shortcut --> same MLP depth for encoder + decoder
        self.encoder_type = encoder_type
        if encoder_type == "mlp":
            self.encoder = MLPEncoder(n_input, n_latent, n_hidden, n_enc_layers, dropout)
        elif encoder_type in ("attention", "hybrid"):
            self.encoder = TokenAttentionEncoder(
                n_input, n_latent, space=space, n_tokens=n_tokens, d_model=d_model, n_heads=n_heads,
                n_attn_layers=n_attn_layers, attn_dropout=attn_dropout, ff_mult=ff_mult, hybrid=encoder_type == "hybrid",
                n_hidden=n_hidden, n_layers=n_enc_layers, dropout=dropout,
            )
        else:
            raise ValueError(f"encoder_type must be mlp / attention / hybrid, got {encoder_type}")
        self.decoder = Decoder(n_latent, n_input, n_hidden, n_dec_layers, dropout)

    def encode(self, x, return_attn: bool = False):
        return self.encoder(x, return_attn)

    @staticmethod
    def reparameterize(mu, logvar):
        #z = mu + sigma * eps
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        #train: encode --> sample z --> decode z
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, x_hat, mu, logvar):
        #scGen loss: mean over cells of 0.5 * SSE + 0.5 * kl_weight * KL(q(z|x) || N(0, I))
        rl = ((x - x_hat) ** 2).sum(1)
        kld = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar).sum(1)
        loss = (0.5 * rl + 0.5 * self.kl_weight * kld).mean()
        return loss, rl.mean(), kld.mean()

    @torch.no_grad()
    def get_latent(self, x, batch_size: int = 256):
        #inference uses the posterior mean, as in scGen
        self.eval()
        dev = next(self.parameters()).device
        out = [self.encode(x[i:i + batch_size].to(dev))[0].cpu() for i in range(0, len(x), batch_size)]
        return torch.cat(out)

    @torch.no_grad()
    def decode_latent(self, z, batch_size: int = 1024):
        self.eval()
        dev = next(self.parameters()).device
        out = [self.decode(z[i:i + batch_size].to(dev)).cpu() for i in range(0, len(z), batch_size)]
        return torch.cat(out)

    @torch.no_grad()
    def get_attention(self, x, batch_size: int = 128):
        #per-cell attention maps --> (N, layers, heads, T+1, T+1), index 0 is [CLS]
        if self.encoder_type == "mlp":
            raise ValueError("mlp encoder has no attention")
        self.eval()
        dev = next(self.parameters()).device
        out = [self.encode(x[i:i + batch_size].to(dev), return_attn=True)[2].cpu() for i in range(0, len(x), batch_size)]
        return torch.cat(out)
