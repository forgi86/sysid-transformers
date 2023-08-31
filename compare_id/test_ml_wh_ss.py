from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
from dataset import LinearDynamicalDataset, WHDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import metrics
from wh_statespace import WHStateSpace
import torchid.ss.dt.estimators as estimators
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset


cuda_device = "cuda:0"
no_cuda = True
threads = 3
batch_size = 32

if __name__ == "__main__":

    # Configure compute
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'  # for later use in torch.autocast

    # Fix all random sources to make script fully reproducible
    torch.manual_seed(420)
    np.random.seed(430)
    system_seed = 430  # Controls the system generation
    data_seed = 0  # Controls the input generation

    # Load out file
    out_dir = Path("../out")
    exp_data = torch.load(out_dir / "ckpt_sim_wh.pt", map_location=device)
    cfg = exp_data["cfg"]

    # Derived quantities
    seq_len = cfg.seq_len_ctx + cfg.seq_len_new

    # Create data loader
    test_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=seq_len,
                        system_seed=system_seed, data_seed=data_seed, fixed_system=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=threads)

    batch_y, batch_u = next(iter(test_dl))
    batch_y = batch_y.to(device)
    batch_u = batch_u.to(device)
    with torch.no_grad():
        batch_y_ctx = batch_y[:, :cfg.seq_len_ctx, :]
        batch_u_ctx = batch_u[:, :cfg.seq_len_ctx, :]
        batch_y_new = batch_y[:, cfg.seq_len_ctx:, :]
        batch_u_new = batch_u[:, cfg.seq_len_ctx:, :]

    #%%
    seq_idx = 0
    #seq_idx = 5
    noise_std = 0.0
    u_train = batch_u_ctx[[seq_idx], :, :]
    u_full = batch_u[[seq_idx], :, :]
    u_new = batch_u_new[[seq_idx], :, :]

    y_train = batch_y_ctx[[seq_idx], :, :]
    y_full = batch_y[[seq_idx], :, :]
    y_new = batch_y_new[[seq_idx], :, :]

    y_train = y_train.clone() + noise_std*torch.randn(u_train.shape)


    #%% Model structure
    seq_fit_len = 80
    seq_est_len = 20
    n_x = 5
    model = WHStateSpace(n_x).to(device)
    estimator = estimators.FeedForwardStateEstimator(n_u=1, n_y=1, n_x=2*n_x,
                                                     hidden_size=16,
                                                     seq_len=seq_est_len,
                                                     batch_first=True)

    train_data = SubsequenceDataset(u_train[0], y_train[0], subseq_len=seq_est_len+seq_fit_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    lr = 2e-4
    num_iter = 10_000  # ADAM iterations 20000

    # In[Setup optimizer]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # In[Train]
    LOSS = []
    start_time = time.time()
    itr = 0
    msg_freq = 100
    for epoch in range(10_000):
        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_u_est = batch_u[:, :seq_est_len]
            batch_y_est = batch_y[:, :seq_est_len]
            batch_x0 = estimator(batch_u_est, batch_y_est)

            batch_u_fit = batch_u[:, seq_est_len:]
            batch_y_fit = batch_y[:, seq_est_len:]

            # Simulate
            batch_y_sim = model(batch_u_fit, batch_x0)

            # Compute fit loss
            err_fit = batch_y_fit[:, :, :] - batch_y_sim[:, :, :]
            loss = torch.mean(err_fit ** 2)

            # Backward pass
            loss.backward()
            optimizer.step()

            LOSS.append(loss.item())
            if itr % msg_freq == 0:
                with torch.no_grad():
                    RMSE = torch.sqrt(loss)
                print(f'Iter {itr} | Fit Loss {loss:.6f} | RMSE:{RMSE:.4f}')
            itr += 1

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%%
    x0 = torch.zeros(1, 2*n_x, device=u_full.device, dtype=u_full.dtype)
    with torch.no_grad():
        y_sim_full = model(u_full, x0)
    y_sim_new = y_sim_full[:, cfg.seq_len_ctx:, :]

    plt.plot(y_full[0], 'k')
    plt.plot(y_sim_full[0], 'b')
    plt.grid()
    plt.axvline(cfg.seq_len_ctx, color='red')
