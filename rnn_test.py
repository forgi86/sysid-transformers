
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
    # device = torch.device("cuda")
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


nx = 5
nu = 1
ny = 1
seq_length = 300

criterion = nn.MSELoss()

model = Model(input_size=nu + ny, output_size=ny, hidden_dim=256, n_layers=4)
model.load_state_dict(torch.load('trained_models/rnn_model20000', map_location=torch.device('cpu')))
model.eval()

test_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, normalize=True)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)

model.eval()
with torch.no_grad():
    for itr, (batch_y, batch_u) in enumerate(test_dl):

        input_seq = batch_u

        init_sequence = 50
        start_zero = torch.zeros([1, 1, ny])
        input_y = batch_y[:, 0:init_sequence - 1, :]
        input_y = torch.cat((start_zero, input_y), dim=1)
        zeros = torch.zeros([1, seq_length - init_sequence, ny])
        input_y = torch.cat((input_y, zeros), dim=1)

        # zeros = torch.zeros([1, seq_length, ny])
        input_seq = torch.cat((input_seq, input_y), dim=2)

        target_seq = batch_y[:, :, 0]
        tgt = target_seq.view(-1).float()

        #output, hidden = model(input_seq)

        results = []
        total_loss = 0
        for i in range(init_sequence, seq_length+1):
            seq = input_seq[:, 0:i, :]
            output, hidden = model(seq)
            results.append(output[-1][0].detach().numpy())
            if i == seq_length:
                break
            loss = criterion(output[-1][0], tgt[i])
            total_loss += loss
            input_seq[:, i, :] = tgt[i]
            # input_seq[:, i, :] = output[-1][0]

        loss = criterion(output[:, 0], tgt)
        out = output[:, 0].detach().numpy()

        # print(tgt)
        # print(out)

        # loss with initial sequence
        print(loss)
        # loss without initial sequence
        print(total_loss / (seq_length-init_sequence))

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].set_title("Input 1")
        ax[0].plot(batch_u[0][:, 0])
        # ax[1].set_title("Input 2")
        # ax[1].plot(batch_u[0][:, 1])
        ax[1].set_title("Output")
        ax[1].plot(tgt)
        ax[1].plot(out, c='red')

        plt.show()