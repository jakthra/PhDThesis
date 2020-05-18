import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class stnn(nn.Module):
    def __init__(self, nx, nt, nd, nz, dropout=0.01):
        super(stnn, self).__init__()
        self.nx = nx # X dim
        self.nt = nt # T dim
        self.nd = nd # Channels
        self.nz = nz # Z dim (decoder size)

        self.factors = nn.Parameter(torch.Tensor(nx, nt, nz))
        self.decoder = nn.Linear(nz, nd, bias=False)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._init_weights()

    def dec_closure(self, x_idx, t_idx):
        z_inf = self.drop(self.factors[x_idx, t_idx])
        z_inf = self.relu(z_inf)
        x_rec = self.decoder(z_inf)
        
        return x_rec

    def factors_parameters(self):
        yield self.factors

    def _init_weights(self, periode=1):
        initrange = 2
        timesteps = torch.arange(self.factors.size(0)).long()
        for t in range(periode):
            idx = timesteps % periode == t
            idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
            init = torch.Tensor(self.nx, self.nz).uniform_(-initrange, initrange).repeat(idx.sum().item(), 1, 1)
        self.factors.data.masked_scatter_(idx_data, init.view(-1))