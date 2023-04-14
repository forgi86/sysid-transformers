import time
import torch
import numpy as np
from dataset import LinearDynamicalDataset
from torch.utils.data import DataLoader
from model_ts import GPTConfig, GPT
import argparse

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # arguments

    nx = 3
    nu = 1
    ny = 1
    seq_len = 100
    max_iter = 10_000

    block_size = seq_len
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.0
    bias = False

    weight_decay = 1e-1
    learning_rate = 6e-4  # max learning rate
    beta1 = 0.9
    beta2 = 0.95

    train_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_len)
    train_dl = DataLoader(train_ds, batch_size=32, num_workers=4)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_y=1, n_u=1, block_size=block_size,
                      bias=bias, vocab_size=200, dropout=dropout)  # start with model_args from command line
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    time_start = time.time()
    LOSS = []
    for itr, (batch_y, batch_u) in enumerate(train_dl):
        #pass  # do something instead...
        optimizer.zero_grad()
        batch_y_pred, loss = model(batch_u, batch_y)
        loss.backward()
        LOSS.append(loss.item())
        print(f"{itr=} {loss=:.2f}")
        if itr == max_iter-1:
            break
        optimizer.step()

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    #batch_yu = torch.cat((batch_y, batch_u), dim=-1)


