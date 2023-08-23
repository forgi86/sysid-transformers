"""
Implementation of an encoder-decoder Transformer model for dynamical systems. Uses as much as possible the plain
pytorch implementation.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    seq_len_ctx: int = 256
    seq_len_new: int = 128
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_u: int = 1
    n_y: int = 1
    dropout: float = 0.0


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_wte = nn.Linear(config.n_u + config.n_y, config.n_embd)  # we process continuous data
        self.encoder_wpe = nn.Embedding(config.seq_len_ctx, config.n_embd)
#        self.decoder_wte = nn.Linear(config.n_u, config.n_embd) # decoder processes u only (output-error model)
        self.decoder_wte = nn.Linear(config.n_u + config.n_y, config.n_embd)
        self.decoder_wpe = nn.Embedding(config.seq_len_new, config.n_embd)
        self.transformer = nn.Transformer(d_model=config.n_embd, nhead=config.n_head,
                                          num_encoder_layers=config.n_layer,
                                          num_decoder_layers=config.n_layer,
                                          batch_first=True)
        self.lm_head = nn.Linear(config.n_embd, config.n_y)


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
    

    def embed_ctx(self, y, u):
        device = u.device
        b, t, nu = u.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        uy = torch.cat((u, y), dim=-1)
        tok_emb = self.encoder_wte(uy)
        pos_emb = self.encoder_wpe(pos)
        src = tok_emb + pos_emb  # perhaps dropout of this?
        return src
    
    def embed_new(self, y_new, u_new):
        device = u_new.device
        b, t_new, nu = u_new.shape
        pos_new = torch.arange(0, t_new, dtype=torch.long, device=device).unsqueeze(0)       
        #tok_emb_new = self.decoder_wte(u_new)

        uy_new = torch.cat((u_new, y_new), dim=-1)
        tok_emb_new = self.decoder_wte(uy_new)

        pos_emb_new = self.decoder_wpe(pos_new)
        tgt = tok_emb_new + pos_emb_new
        return tgt


    def forward(self, y, u, y_new, u_new):

        #device = u.device

        #b, t, nu = u.shape
        #b, t_new, nuu = u_new.shape

        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        #uy = torch.cat((u, y), dim=-1)
        #tok_emb = self.encoder_wte(uy)
        #pos_emb = self.encoder_wpe(pos)
        #src = tok_emb + pos_emb  # perhaps dropout of this?
        src = self.embed_ctx(y, u)

        #pos_new = torch.arange(0, t_new, dtype=torch.long, device=device).unsqueeze(0)       
        #tok_emb_new = self.decoder_wte(u_new)

        #uy_new = torch.cat((u_new, y_new), dim=-1)
        #tok_emb_new = self.decoder_wte(uy_new)

        #pos_emb_new = self.decoder_wpe(pos_new)
        #tgt = tok_emb_new + pos_emb_new
        tgt = self.embed_new(y_new, u_new)
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
    #batch_yu = torch.cat((batch_y, batch_u), dim=-1)
    batch_u_new = torch.randn((batch_size, seq_len_new, n_u))
    batch_y_new = torch.randn((batch_size, seq_len_new, n_y))

    batch_y_new_sim = model(batch_y, batch_u, batch_y_new, batch_u_new)
