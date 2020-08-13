import torch 
import torch.nn as nn
from torchvision.models import resnet18

def create_mask(input):
    zeros = torch.zeros_like(input)
    ones = torch.ones_like(input)
    bs, d = input.size()
    condition = torch.empty_like(input)
    for i in range(bs):
        for j in range(d):
            condition[i][j] = i + j
    mask = torch.where(condition % 2 == 0, ones, zeros)
    return mask

def reverse_mask(mask):
    new_mask = torch.ones_like(mask)
    return new_mask - mask

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCouplingLayer, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 2*input_dim))

    def forward(self, input, binary_mask, reverse=False):
        x0 = torch.mul(input, binary_mask)
        st = self.net(x0)
        # rescale s with tanh and scale factor
        s, t = torch.chunk(st, 2, dim=1)
        s = torch.mul(1-binary_mask, torch.tanh(s))
        t = torch.mul(1-binary_mask, t)
        if reverse:
            # FROM Z TO X
            tmp = torch.mul(input-t, torch.exp(-1.*s))
            output = x0 + torch.mul(1-binary_mask, tmp)
            return output, 0
        else: 
            # FROM X TO Z
            tmp = torch.mul(input, torch.exp(s)) + t
            output = x0 + torch.mul(tmp, 1-binary_mask)
            log_det = s.sum(-1)
        return output, log_det
        

class Net(nn.Module):
    def __init__(self, N, input_dim, hidden_dim):
        super(Net, self).__init__()
        self.n = N
        self.layers = nn.ModuleList([AffineCouplingLayer(input_dim=input_dim, hidden_dim=hidden_dim) for _ in range(self.n)])

    def forward(self, input, reverse=False):
        # stack 3 layers with alternating checkboard pattern.
        binary_mask = create_mask(input)
        log_det_loss = 0 
        z = input
        for idx in range(self.n):
            if (idx+1) % 2 == 0:
                z, log_det = self.layers[idx](z, reverse_mask(binary_mask), reverse)
            else:
                z, log_det = self.layers[idx](z, binary_mask, reverse)
            log_det_loss += log_det
        return z, log_det_loss