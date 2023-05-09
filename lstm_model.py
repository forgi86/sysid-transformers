import torch
from torch import nn

class LSTModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device=None):
        super(LSTModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)



    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)

        #out = self.dropout(out)

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden