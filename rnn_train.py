import torch
from torch import nn
from dataset import LinearDynamicalDataset, WHDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lstm_model import LSTModel
from rnn_model import RNNModel
import math


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    #device = torch.device("cuda")
    device = torch.device(f'cuda:{2}')
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# arguments
nx = 5
nu = 1
ny = 1
seq_length = 500
max_iters = 1000000
batch_size = 64

train_ds = WHDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length)
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=2)

# Instantiate the model with hyperparameters
model = LSTModel(input_size=nu + ny, output_size=ny, hidden_dim=512, n_layers=4)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 1
lr = 0.0001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []

model.train()
for itr, (batch_y, batch_u) in enumerate(train_dl):

    if itr % 200000 == 0:
        torch.save(model.state_dict(), 'trained_models/lstm_wh_model_'+str(itr))
        np.save('trained_models/lstm_wh_losses_'+str(itr), np.array(losses))
    if itr >= max_iters:
        break

    # fig, ax = plt.subplots(3, 1, sharex=True)
    # ax[0].set_title("Input 1")
    # ax[0].plot(batch_u[0][:, 0])
    # ax[1].set_title("Input 2")
    # ax[1].plot(batch_u[0][:, 1])
    # ax[2].set_title("Output")
    # ax[2].plot(batch_y[0])
    #
    # plt.show()

    input_seq = batch_u
    input_y = batch_y[:, 0:seq_length-1, :]
    zeros = torch.zeros([batch_size, 1, ny])
    input_y = torch.cat((zeros, input_y), dim=1)
    input_seq = torch.cat((input_seq, input_y), dim=2).to(device)

    target_seq = batch_y.to(device)

    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        #input_seq.to(device)
        output, hidden = model(input_seq)
        #tgt = target_seq.view(-1).float()
        loss = criterion(output, target_seq)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        #if epoch % 10 == 0:
        print('Itr: {}, Epoch: {}/{}.............'.format(itr, epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        losses.append(loss.item())

torch.save(model.state_dict(), 'trained_models/lstm_wh_model')