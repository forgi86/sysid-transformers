import torch
from torch import nn
from dataset import LinearDynamicalDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lstm_model import LSTModel
from rnn_model import RNNModel

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


nx = 5
nu = 1
ny = 1
seq_length = 300

criterion = nn.MSELoss()

model = RNNModel(input_size=nu + ny, output_size=ny, hidden_dim=512, n_layers=4)
model.load_state_dict(torch.load('trained_models/rnn_model300000', map_location=torch.device('cpu')))
model.eval()

test_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length, normalize=True)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)

model.eval()
with torch.no_grad():
    for itr, (batch_y, batch_u) in enumerate(test_dl):

        input_seq = batch_u

        init_sequence = 50
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
            #input_seq[0][i+1][ny] = batch_y[0][i][0]
            input_seq[0][i+1][ny] = output[0][-1][0]

        loss = criterion(output, batch_y)
        out = output[0, :, 0].detach().numpy()

        print("loss with initial sequence")
        print(loss)
        print("loss without initial sequence")
        print(total_loss / (seq_length-init_sequence))

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].set_title("Input 1")
        ax[0].plot(batch_u[0][:, 0])
        # ax[1].set_title("Input 2")
        # ax[1].plot(batch_u[0][:, 1])
        ax[1].set_title("Output")
        ax[1].plot(batch_y[0, :, 0].detach().numpy())
        ax[1].plot(out, c='red')

        plt.show()
