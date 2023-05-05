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
    #device = torch.device("cuda")
    device = torch.device(f'cuda:{2}')
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
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)



    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(out)

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


# arguments
nx = 5
nu = 1
ny = 1
seq_length = 300
max_iters = 50000
batch_size = 64

train_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, normalize=True)
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4)

# Instantiate the model with hyperparameters
model = Model(input_size=nu + ny, output_size=ny, hidden_dim=256, n_layers=4)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define hyperparameters
n_epochs = 1
lr = 0.0001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for itr, (batch_y, batch_u) in enumerate(train_dl):

    if itr % 10000 == 0:
        torch.save(model.state_dict(), 'trained_models/rnn_model'+str(itr))
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

    target_seq = batch_y[:, :, 0].to(device)

    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        #input_seq.to(device)
        output, hidden = model(input_seq)
        tgt = target_seq.view(-1).float()
        loss = criterion(output[:, 0], tgt)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        #if epoch % 10 == 0:
        print('Itr: {}, Epoch: {}/{}.............'.format(itr, epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

torch.save(model.state_dict(), 'trained_models/rnn_model')