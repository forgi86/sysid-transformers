import torch
from torch import nn

class LSTModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x):

        out, hidden = self.lstm(x)

        out = self.fc(out)

        return out, hidden