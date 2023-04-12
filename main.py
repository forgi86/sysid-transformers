import time
import torch
import numpy as np
from dataset import LinearDynamicalDataset
from torch.utils.data import DataLoader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # arguments
    nx = 3
    nu = 1
    ny = 1
    seq_len = 500
    max_iter = 100

    train_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_len)
    train_dl = DataLoader(train_ds, batch_size=32, num_workers=4)

    time_start = time.time()
    for itr, (batch_y, batch_u) in enumerate(train_dl):
        pass  # do something instead...
        #print(itr)
        if itr == max_iter-1:
            break

    time_loop = time.time() - time_start
    print(f"{time_loop=:.2f} seconds.")
