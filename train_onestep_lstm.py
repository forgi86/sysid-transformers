from pathlib import Path
import torch
from lstm_onestep import LSTModel
from torch import nn
from dataset import LinearDynamicalDataset, WHDataset
from lstm_onestep import LSTModel
from torch.utils.data import DataLoader
import time

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    #device = torch.device("cuda")
    device = torch.device(f'cuda:{0}')
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

model_dir = "out"


# Define hyperparameters
nx = 10
nu = 1
ny = 1
seq_length = 500
iterations = 1000000
batch_size = 64
lr = 0.0001
wh_dataset = False

# Create out dir
model_dir = Path(model_dir)
model_dir.mkdir(exist_ok=True)
out_file = "ckpt_onestep_wh_lstm" if wh_dataset else "ckpt_onestep_lin_lstm"

if wh_dataset:
    train_ds = WHDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length)
else:
    train_ds = LinearDynamicalDataset(nx=nx, nu=nu, ny=ny, seq_len=seq_length)
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=2)

# Instantiate the model with hyperparameters
model = LSTModel(input_size=nu + ny, output_size=ny, hidden_dim=512, n_layers=4)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []

time_start = time.time()

model.train()
for itr, (batch_y, batch_u) in enumerate(train_dl):

    if itr % int((iterations/5)) == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': itr,
            'train_time': time.time() - time_start,
            'training_loss': losses
        }
        torch.save(checkpoint, model_dir / f"{out_file}.pt")

    if itr >= iterations:
        break

    input_seq = batch_u
    input_y = batch_y[:, 0:seq_length-1, :]
    zeros = torch.zeros([batch_size, 1, ny])
    input_y = torch.cat((zeros, input_y), dim=1)
    input_seq = torch.cat((input_seq, input_y), dim=2).to(device)

    target_seq = batch_y.to(device)

    optimizer.zero_grad()  # Clears existing gradients from previous epoch

    output, hidden = model(input_seq)

    loss = criterion(output, target_seq)
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    print('Itr: {}/{}.............'.format(itr, iterations), end=' ')
    print("Loss: {:.4f}".format(loss.item()))
    losses.append(loss.item())

checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iterations,
            'train_time': time.time() - time_start,
            'training_loss': losses
        }

torch.save(checkpoint, model_dir / f"{out_file}_last.pt")
