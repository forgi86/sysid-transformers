"""
Implementation of the Transformer models for dynamical systems. Derived from Karpathy's nanoGPT
https://github.com/karpathy/nanoGPT/
"""

import math
from dataclasses import dataclass
import torch.nn as nn
import torch
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
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=True, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(x, x, x, is_causal=False)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class CrossAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, bias=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads,
                                         bias=bias, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.causal = causal

    def forward(self, x, mem):
        x = self.mha(x, mem, mem, is_causal=self.causal)[0]
        #y = self.resid_dropout(self.c_proj(x))
        y = self.resid_dropout(x)  # projection already in mha!
        return y


class MLP(nn.Module):

    def __init__(self, d_model, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_head, dropout=dropout, causal=False, bias=bias) # encoder is never causal
        
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.self_attn = SelfAttention(d_model, n_heads,
                                       dropout=dropout, causal=True, bias=bias)
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads,
                                         dropout=dropout, causal=False, bias=bias)
        self.ln_3 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model)

    def forward(self, x, mem):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), mem)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model, bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # final layer normalization
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model, bias)

    def forward(self, x, mem):
        for block in self.blocks:
            x = block(x, mem)
        x = self.ln_f(x)  # final layer normalization
        return x


class TSTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)
        self.decoder = TransformerDecoder(config.n_embd, config.n_head, config.n_layer,
                                          dropout=config.dropout, bias=config.bias)

        self.encoder_wte = nn.Linear(config.n_u + config.n_y, config.n_embd)
        self.encoder_wpe = nn.Embedding(config.seq_len_ctx, config.n_embd)
        self.decoder_wte = nn.Linear(config.n_u + config.n_y, config.n_embd)
        self.decoder_wpe = nn.Embedding(config.seq_len_new, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.n_y, bias=True)  # keep bias here maybe?

    def embed_ctx(self, y, u):
        device = u.device
        b, t, nu = u.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        yu = torch.cat((y, u), dim=-1)
        tok_emb = self.encoder_wte(yu)
        pos_emb = self.encoder_wpe(pos)
        src = tok_emb + pos_emb  # perhaps dropout of this?
        return src

    def embed_new(self, y_new, u_new):
        device = u_new.device
        b, t_new, nu = u_new.shape
        pos_new = torch.arange(0, t_new, dtype=torch.long, device=device).unsqueeze(0)

        yu_new = torch.cat((y_new, u_new), dim=-1)
        tok_emb_new = self.decoder_wte(yu_new)
        pos_emb_new = self.decoder_wpe(pos_new)
        tgt = tok_emb_new + pos_emb_new
        return tgt

    def forward(self, y, u, y_new, u_new):
        src = self.embed_ctx(y, u)  # perhaps dropout of this?
        tgt = self.embed_new(y_new, u_new)  # perhaps dropout of this?
        mem = self.encoder(src)
        output = self.decoder(tgt, mem)
        y_new_sim = self.lm_head(output)
        return y_new_sim

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True #'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


if __name__ == "__main__":
    batch_size = 8
    seq_len_ctx = 128
    seq_len_new = 64
    n_u = 2
    n_y = 3

    cfg = Config(seq_len_ctx=seq_len_ctx, seq_len_new=seq_len_new, n_u=n_u, n_y=n_y)
    model = TSTransformer(cfg)

    batch_y = torch.randn((batch_size, seq_len_ctx, n_y))
    batch_u = torch.randn((batch_size, seq_len_ctx, n_u))

    batch_y_ctx = batch_y[:, :cfg.seq_len_ctx, :]
    batch_u_ctx = batch_u[:, :cfg.seq_len_ctx, :]

    batch_y_new = batch_y[:, cfg.seq_len_ctx:, :]
    batch_u_new = batch_u[:, cfg.seq_len_ctx:, :]

    model.eval()
    batch_y_new_sim = model(batch_y, batch_u, batch_y_new, batch_u_new)
