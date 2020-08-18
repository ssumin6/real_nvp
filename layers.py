import torch 
import torch.nn as nn
from torchvision.models import resnet18

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(AffineCouplingLayer, self).__init__()
        self.mask = mask
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 2*input_dim))

    def forward(self, input, reverse=False):
        x0 = torch.mul(input, self.mask)
        st = self.net(x0)
        # rescale s with tanh and scale factor
        s, t = torch.chunk(st, 2, dim=1)
        s = torch.mul(1-self.mask, torch.tanh(s))
        t = torch.mul(1-self.mask, t)
        if reverse:
            # FROM Z TO X
            tmp = torch.mul(input-t, torch.exp(-s))
            output = x0 + torch.mul(1-self.mask, tmp)
            log_det = -s.sum(-1)
        else: 
            # FROM X TO Z
            tmp = torch.mul(input, torch.exp(s)) + t
            output = x0 + torch.mul(1-self.mask, tmp)
            log_det = s.sum(-1)
        return output, log_det 

class Net(nn.Module):
    def __init__(self, N, input_dim, hidden_dim, device):
        super(Net, self).__init__()
        self.n = N
        self.device = device
        self.masks = torch.Tensor([[0, 1], [1, 0], [0, 1], [1, 0]]).to(self.device)
        self.layers = nn.ModuleList([AffineCouplingLayer(input_dim=input_dim, hidden_dim=hidden_dim, mask=self.masks[i]) for i in range(self.n)])

    def forward(self, input, reverse=False):
        # stack 3 layers with alternating checkboard pattern.
        log_det_loss = torch.zeros(input.size()[0]).to(self.device)
        z = input
        index_range = range(self.n) if not reverse else range(self.n-1, -1 , -1)
        for idx in index_range:
            z, log_det = self.layers[idx](z, reverse)
            log_det_loss += log_det
        return z, log_det_loss