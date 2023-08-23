"""
Implementation of the Transformer models for dynamical systems. Derived from Karpathy's nanoGPT
https://github.com/karpathy/nanoGPT/
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    seq_len_ctx: int = 128
    seq_len_new: int = 128
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_u: int = 1
    n_y: int = 1
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_wte = nn.Linear(config.n_u + config.n_y, config.n_embd)  # we process continuous data
        self.encoder_wpe = nn.Embedding(config.seq_len_ctx, config.n_embd)
        self.decoder_wte = nn.Linear(config.n_u + config.n_y, config.n_embd)  # we process continuous data
        self.decoder_wpe = nn.Embedding(config.seq_len_new, config.n_embd)
        self.transformer = nn.Transformer(d_model=config.n_embd, nhead=config.n_head,
                                          num_encoder_layers=config.n_layer,
                                          num_decoder_layers=config.n_layer,
                                          batch_first=True)
        self.lm_head = nn.Linear(config.n_embd, config.n_y)

    def forward(self, u, y, u_new, y_new):
        device = u.device
        b, t, nu = u.shape
        b, t_new, nuu = u_new.shape

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        uy = torch.cat((u, y), dim=-1)
        tok_emb = self.encoder_wte(uy)
        pos_emb = self.encoder_wpe(pos)
        src = tok_emb + pos_emb  # perhaps dropout of this?

        pos_new = torch.arange(0, t_new, dtype=torch.long, device=device).unsqueeze(0)
        uy_new = torch.cat((u_new, y_new), dim=-1)
        tok_emb_new = self.decoder_wte(uy_new)
        pos_emb_new = self.decoder_wpe(pos_new)
        tgt = tok_emb_new + pos_emb_new

        x = self.transformer(src, tgt)
        y_new_sim = self.lm_head(x)
        return y_new_sim


if __name__ == "__main__":
    batch_size = 8
    seq_len_ctx = 256
    seq_len_new = 128
    n_u = 2
    n_y = 3

    model_cfg = Config(seq_len_ctx=seq_len_ctx, seq_len_new=seq_len_new, n_u=n_u, n_y=n_y)
    model = TransformerModel(model_cfg)

    batch_u = torch.randn((batch_size, seq_len_ctx, n_u))
    batch_y = torch.randn((batch_size, seq_len_ctx, n_y))
    batch_yu = torch.cat((batch_y, batch_u), dim=-1)
    batch_u_new = torch.randn((batch_size, seq_len_new, n_u))
    batch_y_new = torch.randn((batch_size, seq_len_new, n_y))

    batch_y_new_sim = model(batch_u, batch_y, batch_u_new, batch_y_new)
