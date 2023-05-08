from pathlib import Path
import time
import torch
import numpy as np
import math
from dataset import LinearDynamicalDataset
from torch.utils.data import DataLoader
from model import GPTConfig, GPT
import tqdm
import argparse


if __name__ == '__main__':

    # Overall settings
    out_dir = "out"

    # System settings
    nx = 10
    nu = 1
    ny = 1
    seq_len = 400

    # Transformer settings
    block_size = seq_len
    n_layer = 8
    n_head = 4
    n_embd = 128
    dropout = 0.0
    bias = False

    # Optimization settings
    learning_rate = 6e-4
    decay_lr = True
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    warmup_iters = 2000
    max_iter = 300_000 # 600_000
    lr_decay_iters = max_iter
    min_lr = learning_rate/10.0
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

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # Training loop
    time_start = time.time()
    LOSS = []
    for iter_num, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl)):

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if device_type == "cuda":
            batch_y = batch_y.pin_memory().to(device, non_blocking=True)
            batch_u = batch_u.pin_memory().to(device, non_blocking=True)
        batch_y_pred, loss = model(batch_u, batch_y)
        LOSS.append(loss.item())
        if iter_num % 100 == 0:
            print(f"\n{iter_num=} {loss=:.4f}\n")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter_num == max_iter-1:
            break

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    # Save results
    checkpoint = {
        'model': model.state_dict(),
        'model_args': model_args,
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'train_time': time_loop,
        'LOSS': LOSS,
    }
    torch.save(checkpoint, out_dir/"ckpt.pt")
    


