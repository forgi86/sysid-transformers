import torch
from torch import nn
from dataset import LinearDynamicalDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, dropout=0.15)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        self.droput = nn.Dropout(0.15)


    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)

        out = self.droput(out)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# arguments
nx = 10
nu = 1
ny = 1
seq_length = 100
max_iters = 200
batch_size = 64
train_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, normalize=True)
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0)

# Instantiate the model with hyperparameters
model = Model(input_size=nu+ny, output_size=ny, hidden_dim=256, n_layers=2)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 100
lr = 0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.train()
for itr, (batch_y, batch_u) in enumerate(train_dl):

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
    input_y = batch_y[:, 1:seq_length, :]
    zeros = torch.zeros([batch_size, 1, ny])
    input_y = torch.cat((zeros, input_y), dim=1)
    input_seq = torch.cat((input_seq, input_y), dim=2)

    target_seq = batch_y[:, :, 0]

    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        input_seq.to(device)
        output, hidden = model(input_seq)
        tgt = target_seq.view(-1).float()
        loss = criterion(output[:, 0], tgt)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            print('Itr: {}, Epoch: {}/{}.............'.format(itr, epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))


test_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, normalize=True)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)
model.eval()
for itr, (batch_y, batch_u) in enumerate(test_dl):

    input_seq = batch_u
    zeros = torch.zeros([1, seq_length, ny])
    input_seq = torch.cat((input_seq, zeros), dim=2)

    target_seq = batch_y[:, :, 0]

    output, hidden = model(input_seq)

    results = []
    for i in range(seq_length):
        seq = input_seq[:, 0:i+1, :]
        output, hidden = model(seq)
        results.append(output[-1][0].detach().numpy())
        if i == seq_length-1:
            break
        input_seq[:, i+1, :] = output[-1][0]

    tgt = target_seq.view(-1).float()
    loss = criterion(output[:, 0], tgt)
    out = output[:, 0].detach().numpy()

    print(tgt)
    print(out)
    print(loss)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("Input 1")
    ax[0].plot(batch_u[0][:, 0])
    # ax[1].set_title("Input 2")
    # ax[1].plot(batch_u[0][:, 1])
    ax[1].set_title("Output")
    ax[1].plot(tgt)
    ax[1].plot(out, c='red')

    plt.show()

