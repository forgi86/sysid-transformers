import torch

from torchid.ss.dt.simulator import StateSpaceSimulator
import torchid.ss.dt.models as models


n_hidden = 32
n_x = 10
f_xu = models.NeuralLinStateUpdate(n_x, 1, hidden_size=n_hidden)#.to(device)
g_x = models.NeuralLinOutput(n_x, 1, hidden_size=n_hidden)#.to(device)
model = StateSpaceSimulator(f_xu, g_x, batch_first=True)#.to(device)

batch_size = 1
seq_len = 400
n_u = 1
u_train = torch.randn(batch_size, seq_len, n_u)
x0 = torch.zeros(batch_size, n_x, device=u_train.device, dtype=u_train.dtype)