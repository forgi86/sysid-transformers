import torch
from torch import nn
from dataset import LinearDynamicalDataset, WHDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lstm_model import LSTModel
from model import GPTConfig, GPT
from rnn_model import RNNModel
import numpy as np
from torch.nn import functional as F
import math

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device(f'cuda:{2}')
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

exp_data = torch.load("trained_models/ckpt_small_wh.pt", map_location=device)

seq_length = 300 #exp_data["cfg"].seq_len
nx = exp_data["cfg"].nx
nu = 1
ny = 1
show_plots = False
model_args = exp_data["model_args"]
gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

state_dict = exp_data["model"]
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

test_ds = WHDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, system_seed=42, data_seed=42)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)

model.eval()
losses_one_step = []
losses_sim = []
with torch.no_grad():
    for itr, (batch_y, batch_u) in enumerate(test_dl):

        if itr == 100:
            break

        output_one_step, loss_one_step = model(batch_u, batch_y)

        print(loss_one_step)
        losses_one_step.append(loss_one_step)

        if show_plots:
            fig, ax = plt.subplots(nu+1, 1, sharex=True)

            for i in range(nu):
                ax[i].set_title(f"Input {i + 1}")
                ax[i].plot(batch_u[0][:, i])

            ax[nu].set_title("Output")
            ax[nu].plot(batch_y[0, :, 0].detach().numpy())
            ax[nu].plot(output_one_step[0, :, 0].detach().numpy(), c='red')

            plt.show()

        sim_start = 100

        batch_y_sim = torch.zeros_like(batch_y)
        batch_y_sim[:, :sim_start, :] = batch_y[:, :sim_start, :]

        for idx in range(sim_start, seq_length):
            batch_y_t, _ = model(batch_u[:, :idx, :], batch_y_sim[:, :idx, :], compute_loss=False)
            batch_y_sim[:, [idx], :] = batch_y_t

        loss_sim = F.mse_loss(batch_y[:, sim_start:, :], batch_y_sim[:, sim_start:, :])

        print(loss_sim)
        losses_sim.append(loss_sim)

        if show_plots:
            fig, ax = plt.subplots(nu + 1, 1, sharex=True)

            for i in range(nu):
                ax[i].set_title(f"Input {i + 1}")
                ax[i].plot(batch_u[0][:, i])

            ax[nu].set_title("Output")
            ax[nu].plot(batch_y[0, :, 0].detach().numpy())
            ax[nu].plot(batch_y_sim[0, :, 0].detach().numpy(), c='red')

            plt.show()

losses_one_step = np.array(losses_one_step)
losses_sim = np.array(losses_sim)

np.save('transformer_wh_losses_one_step', losses_one_step)
np.save('transformer_wh_losses_sim', losses_sim)
