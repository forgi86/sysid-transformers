import time
import torch
import numpy as np
from torchid.dynonet.module.lti import SisoLinearDynamicalOperator
from torchid.dynonet.module.static import SisoStaticNonLinearity
import metrics
import scipy.io as sio
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')
    parser.add_argument('--noise-std', type=float, default=0.0, metavar='N', help='Added noise in the context')
    cfg = parser.parse_args()
    noise_std = cfg.noise_std

    # Fix all random sources to make script fully reproducible
    torch.manual_seed(420)
    np.random.seed(430)

    seq_len_ctx = 400

    # Configure compute
    torch.set_num_threads(10)


    out_file = f"batch_y_sim_full_rep_{noise_std:.1f}.pt"

    batch_size = 32
    wh_data = sio.loadmat("wh_data.mat")
    batch_u = torch.tensor(wh_data["batch_u"][:batch_size])
    batch_y = torch.tensor(wh_data["batch_y"][:batch_size])
    batch_size = batch_u.shape[0]

    # Prepare data
    batch_y_ctx = batch_y[:, :seq_len_ctx, :]
    batch_u_ctx = batch_u[:, :seq_len_ctx, :]
    batch_y_new = batch_y[:, seq_len_ctx:, :]
    batch_u_new = batch_u[:, seq_len_ctx:, :]


    #%% Training function
    def train_model(u, y, log=False, num_iter=100_000):
        
        # Model parameters
        order = 5
        
        # Learning parameters
        #num_iter = 100_000
        #num_iter_max = 1_000_000
        n_skip = 100
        lr = 1e-3
        
        # Setup dynoNet
        G1 = SisoLinearDynamicalOperator(order, order, n_k=1) 
        F_nl = SisoStaticNonLinearity(n_hidden=32, activation='tanh')
        G2 = SisoLinearDynamicalOperator(order+1, order)
        model = torch.nn.Sequential(G1, F_nl, G2)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss function
        def closure():
            optimizer.zero_grad()

            # Simulate
            y_hat = model(u)

            # Compute fit loss
            err_fit = y[:, n_skip:, :] - y_hat[:, n_skip:, :]
            loss = torch.mean(err_fit**2)

            # Backward pass
            loss.backward()
            return loss


        # Training loop
        LOSS = []
        msg_freq = 1000
        start_time = time.time()
        for itr in range(0, num_iter):
                
            loss_train = optimizer.step(closure)

            LOSS.append(loss_train.item())
            if log:
                if itr % msg_freq == 0:
                    with torch.no_grad():
                        RMSE = torch.sqrt(loss_train)
                    print(f'Iter {itr} | Fit Loss {loss_train:.6f} | RMSE:{RMSE:.4f}')
        
        if log:
            train_time = time.time() - start_time
            print(f"\nTrain time: {train_time:.2f}")
        return model

    #%% Actual training here
    time_start = time.time()

    n_rep = 20
    num_iter = 50_000
    y_rep = []
    for rep in range(n_rep):
        print(f"Training repetition {rep}")
        y_sim_full = []
        for seq_idx in range(batch_size):
            #print(f"Fitting sequence {seq_idx+1}")
            
            model = train_model(batch_u_ctx[[seq_idx]],
                                batch_y_ctx[[seq_idx]]  + noise_std * torch.randn(batch_y_ctx[[seq_idx]].shape), num_iter=50_000)
            
            with torch.no_grad():
                y_sim_full.append(
                    model(batch_u[[seq_idx]])
                )       
        batch_y_sim_full = torch.cat(y_sim_full)
        batch_y_sim_new = batch_y_sim_full[:, seq_len_ctx:, :]
        y_rep.append(batch_y_sim_full)

    batch_y_sim_full_rep = torch.stack(y_rep)
    torch.save(batch_y_sim_full_rep, out_file)
    time_train = time.time() - time_start


    #%% Pick up the best repetition
    batch_y_sim_full_rep = torch.load(out_file)
    err_rep = batch_y_sim_full_rep - batch_y # error for all training repetitions R, B, T, C
    err_rep_train = err_rep[:, :seq_len_ctx, :, :] # training error for all repetitions R, B, T, C
    loss_rep = torch.mean(err_rep_train**2, dim=(2, 3)) # loss for all repetitions R, B, L
    idx_best = torch.argmin(torch.nan_to_num(loss_rep, torch.inf),  dim=0) # idx of best training repetition per batch B
    batch_y_sim_full_best = batch_y_sim_full_rep[idx_best, torch.arange(batch_size)] # best training repetition full sequence B, T, C
    batch_y_sim_new = batch_y_sim_full_best[:, seq_len_ctx:, :] # best training repetition test sequence B, T, C

    #%% Stats
    skip = 0
    rmse_ml = metrics.rmse(batch_y_new.numpy(), batch_y_sim_new.numpy(), time_axis=1)
    rmse_ml = np.nan_to_num(rmse_ml, copy=True, nan=np.nanmean(rmse_ml))
    print(f'rmse: {rmse_ml.mean()}')
