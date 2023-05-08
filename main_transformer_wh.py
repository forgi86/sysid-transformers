from pathlib import Path
import time
import torch
import numpy as np
import math
from dataset import WHSmoothDataset
from torch.utils.data import DataLoader
from model_ts import GPTConfig, GPT
import tqdm
import argparse


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='State-space neural network tests')
    # parser.add_argument('--model-dir', type=str, default="out", metavar='S',
    #                     help='Saved model folder')
    # parser.add_argument('--out-file', type=str, default="ckpt", metavar='S',
    #                     help='Saved model name')
    # parser.add_argument('--in-file', type=str, default="ckpt", metavar='S',
    #                     help='Loaded model name (when resuming)')
    # parser.add_argument('--init-from', type=str, default="scratch", metavar='S',
    #                     help='Init either from scratch or from previous checkpoint')
    # args = parser.parse_args()

    # Save/load settings
    model_dir = "out"
    out_file = "ckpt_wh"
    init_from = "scratch"
    #init_from = "resume"
    in_file = "ckpt_wh"

    # System settings
    nx = 10
    nu = 1
    ny = 1
    seq_len = 600

    # Transformer settings
    block_size = seq_len
    n_layer = 12
    n_head = 4
    n_embd = 128
    dropout = 0.0
    bias = False

    # Optimization settings
    learning_rate = 1e-3 #6e-4
    decay_lr = True
    weight_decay = 0.0#1e-1
    beta1 = 0.9
    beta2 = 0.95
    warmup_iters = 10_000
    max_iters = 1_000_000  # 600_000
    lr_decay_iters = max_iters
    min_lr = learning_rate/10.0
    batch_size = 32

    eval_interval = 2000
    eval_iters = 100
    eval_batch_size = 32

    # Compute settings
    cuda_device = "cuda:0"
    no_cuda = False
    threads = 10
    compile = False

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(43)

    # Create out dir
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    # Configure compute
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu' # for later use in torch.autocast
    torch.set_float32_matmul_precision("high")
    #torch._dynamo.config.suppress_errors = True
    #torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    #torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # Create data loader
    train_ds = WHSmoothDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=threads)

    val_ds = WHSmoothDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_len)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, num_workers=threads)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_y=1, n_u=1, block_size=block_size,
                      bias=bias, dropout=dropout)  # start with model_args from command line

    if init_from == "scratch":
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        ckpt_path = model_dir / f"{in_file}.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0

    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])

    def get_lr(iter):
        # 1) linear warmup for warmup_iters steps
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iter > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        loss = 0.0
        for eval_iter, (batch_y, batch_u) in enumerate(val_dl):
            if device_type == "cuda":
                batch_y = batch_y.pin_memory().to(device, non_blocking=True)
                batch_u = batch_u.pin_memory().to(device, non_blocking=True)
            _, loss_iter = model(batch_u, batch_y)
            loss += loss_iter.item()
            if eval_iter == eval_iters:
                break
        loss /= eval_iters
        model.train()
        return loss

    # Training loop
    LOSS_ITR = []
    LOSS_VAL = []
    loss_val = np.nan

    if init_from == "scratch":
        iter_num = 0
        best_val_loss = np.inf
    elif init_from == "resume":
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint['best_val_loss']

    time_start = time.time()
    for iter_num, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl, start=iter_num)):

        if (iter_num % eval_interval == 0) and iter_num > 0:
            loss_val = estimate_loss()
            LOSS_VAL.append(loss_val)
            print(f"\n{iter_num=} {loss_val=:.4f}\n")
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'train_time': time.time() - time_start,
                    'LOSS': LOSS_ITR,
                    'LOSS_VAL': LOSS_VAL,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, model_dir / f"{out_file}.pt")
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if device_type == "cuda":
            batch_y = batch_y.pin_memory().to(device, non_blocking=True)
            batch_u = batch_u.pin_memory().to(device, non_blocking=True)
        batch_y_pred, loss = model(batch_u, batch_y)
        LOSS_ITR.append(loss.item())
        if iter_num % 100 == 0:
            print(f"\n{iter_num=} {loss=:.4f} {loss_val=:.4f}\n")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter_num == max_iters-1:
            break

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'train_time': time.time() - time_start,
        'LOSS': LOSS_ITR,
        'LOSS_VAL': LOSS_VAL,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, model_dir / f"{out_file}_last.pt")
    


