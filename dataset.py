import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim


class LinearDynamicalDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=500, strictly_proper=True, dtype="float32", normalize=True):
        super(LinearDynamicalDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize

    def __iter__(self):
        while True:  # infinite dataset
            # for _ in range(1000):
            sys = control.drss(states=self.nx,
                               inputs=self.nu,
                               outputs=self.ny,
                               strictly_proper=self.strictly_proper)
            u = np.random.randn(self.nu, self.seq_len).astype(self.dtype)  # C, T as python-control wants
            y = control.forced_response(sys, T=None, U=u, X0=0.0)
            u = u.transpose()  # T, C
            y = y.y.transpose().astype(self.dtype)  # T, C
            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            yield torch.tensor(y), torch.tensor(u)


class WHDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, model_seed=None, data_seed=None, **mdlargs):
        super(WHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.model_rng = np.random.default_rng(model_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.mdlargs = mdlargs

    def __iter__(self):

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        n_in = 1
        n_out = 1
        n_hidden = 32
        n_skip = 200

        if self.fixed_system:  # same model at each step, generate only once!
            w1 = self.model_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.model_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.model_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.model_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.model_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=self.strictly_proper,
                               rng=self.model_rng,
                               **self.mdlargs)

            G2 = drss_matrices(states=self.model_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=False,
                               rng=self.model_rng,
                               **self.mdlargs)

        while True:  # infinite dataset

            if not self.fixed_system:  # different model for different instances!
                w1 = self.model_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
                b1 = self.model_rng.normal(size=(1, n_hidden)) * 1.0
                w2 = self.model_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
                b2 = self.model_rng.normal(size=(1, n_out)) * 1.0

                G1 = drss_matrices(states=self.model_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=self.strictly_proper,
                                   rng=self.model_rng,
                                   **self.mdlargs)

                G2 = drss_matrices(states=self.model_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=False,
                                   rng=self.model_rng,
                                   **self.mdlargs)

            #u = np.random.randn(self.seq_len + n_skip, 1)  # input to be improved (filtered noise, multisine, etc)
            u = self.data_rng.normal(size=(self.seq_len + n_skip, 1))

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            yield torch.tensor(y), torch.tensor(u)


class PWHDataset(IterableDataset):
    def __init__(self, nx=50, nu=1, ny=1, nbr=10, seq_len=1024, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32", **mdlargs):
        super(PWHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nbr = nbr
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order
        self.mdlargs = mdlargs

    def __iter__(self):

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        while True:  # infinite dataset
            # for _ in range(1000):

            n_in = 1
            n_out = 1
            n_hidden = 128
            n_skip = 200

            w1 = np.random.randn(n_hidden, n_in) / np.sqrt(n_in) * 1.0
            b1 = np.random.randn(1, n_hidden) * 1.0
            w2 = np.random.randn(n_out, n_hidden) / np.sqrt(n_hidden) * 5/3
            b2 = np.random.randn(1, n_out) * 1.0

            G1 = drss_matrices(states=np.random.randint(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=self.strictly_proper,
                               **self.mdlargs)

            G2 = drss_matrices(states=np.random.randint(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=False,
                               **self.mdlargs)

            # which kind of randomness for u?
            u = np.random.randn(self.seq_len + n_skip, 1)  # input to be improved (filtered noise, multisine, etc)

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            yield torch.tensor(y), torch.tensor(u)

            
if __name__ == "__main__":
    train_ds = WHDataset(nx=2, seq_len=4, mag_range=(0.5, 0.96),
                         phase_range=(0, math.pi / 3),
                         model_seed=42, data_seed=445, fixed_system=False)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=2)
    batch_y, batch_u = next(iter(train_dl))
    batch_y, batch_u = next(iter(train_dl))
    print(batch_u.shape, batch_u.shape)
