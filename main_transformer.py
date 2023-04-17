from pathlib import Path
import time
import torch
import numpy as np
from dataset import LinearDynamicalDataset
from torch.utils.data import DataLoader
from model_ts import GPTConfig, GPT
import tqdm
import argparse


if __name__ == '__main__':

    # Overall settings
    out_dir = "out"

    # System settings
    nx = 10
    nu = 1
    ny = 1
    seq_len = 500

    # Transformer settings
    block_size = seq_len
    n_layer = 8
    n_head = 4
    n_embd = 128
    dropout = 0.0
    bias = False

    # Optimization settings
    weight_decay = 1e-1
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.95
    max_iter = 100_000
    batch_size = 32


    # Compute settings
    cuda_device = "cuda:0"
    no_cuda = False
    threads = 5
    compile = True

    # %% Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(43)

    # Create out dir
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Configure compute
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name  = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu' # for later use in torch.autocast
    torch.set_float32_matmul_precision("high")
    #torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    #torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # Create data loader
    train_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=threads)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_y=1, n_u=1, block_size=block_size,
                      bias=bias, dropout=dropout)  # start with model_args from command line
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    time_start = time.time()
    LOSS = []
    for itr, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl)):

        if device_type == "cuda":
            batch_y = batch_y.pin_memory().to(device, non_blocking=True)
            batch_u = batch_u.pin_memory().to(device, non_blocking=True)
        batch_y_pred, loss = model(batch_u, batch_y)
        LOSS.append(loss.item())
        if itr % 100 == 0:
            print(f"\n{itr=} {loss=:.2f}\n")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if itr == max_iter-1:
            break

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    # Save results
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'train_time': time_loop,
        'LOSS': LOSS,
    }
    torch.save(checkpoint, out_dir/"ckpt.pt")
    


