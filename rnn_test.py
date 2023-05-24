import torch
from torch import nn
from dataset import LinearDynamicalDataset, WHDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lstm_model import LSTModel
from rnn_model import RNNModel
import numpy as np
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


nx = 50
nu = 3
ny = 1
seq_length = 500

criterion = nn.MSELoss()

model = LSTModel(input_size=nu + ny, output_size=ny, hidden_dim=512, n_layers=4)
model.load_state_dict(torch.load('trained_models/lstm_model_50nx_500000', map_location=device))

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

test_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, normalize=True)
#test_ds = WHDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, mag_range=(0.5, 0.97), phase_range=(0, math.pi/2))
test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)

model.eval()
losses = []
with torch.no_grad():
    for itr, (batch_y, batch_u) in enumerate(test_dl):

        if itr == 100:
            break

        input_seq = batch_u

        init_sequence = 100
        start_zero = torch.zeros([1, 1, ny])
        input_y = batch_y[:, 0:init_sequence, :]
        input_y = torch.cat((start_zero, input_y), dim=1)
        zeros = torch.zeros([1, seq_length - init_sequence - 1, ny])
        input_y = torch.cat((input_y, zeros), dim=1)

        # zeros = torch.zeros([1, seq_length, ny])
        input_seq = torch.cat((input_seq, input_y), dim=2)

        results = []
        total_loss = 0
        for i in range(init_sequence, seq_length):
            seq = input_seq[:, 0:i+1, :]
            output, hidden = model(seq)
            results.append(output[0][-1][0].detach().numpy())
            loss = criterion(output[0][-1][0], batch_y[0][i][0])
            total_loss += loss
            if i+1 == seq_length:
                break
            #input_seq[0][i+1][nu] = batch_y[0][i][0]
            input_seq[0][i+1][nu] = output[0][-1][0]

        loss = criterion(output, batch_y)
        out = output[0, :, 0].detach().numpy()

        # print("loss with initial sequence")
        # print(loss)
        print("loss without initial sequence")
        mean_loss = total_loss / (seq_length-init_sequence)
        print(mean_loss)

        losses.append(mean_loss.detach().numpy())

        fig, ax = plt.subplots(nu+1, 1, sharex=True)

        for i in range(nu):
            ax[i].set_title(f"Input {i+1}")
            ax[i].plot(batch_u[0][:, i])

        ax[nu].set_title("Output")
        ax[nu].plot(batch_y[0, :, 0].detach().numpy())
        ax[nu].plot(out, c='red')

        plt.show()

losses = np.array(losses)
print(losses)
print(np.mean(losses))
np.save('lstm_50nx_sim', losses)
