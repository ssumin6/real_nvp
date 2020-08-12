import torch 
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self):
        super(AffineCouplingLayer, self).__init__()
        self.net = nn.Sequential()

    def forward(self, input, binary_mask):
        # FROM X TO Z
        x0 = torch.mul(input, binary_mask)
        st = self.net(x0)
        log_s, t = st[:], st[:]
        tmp = torch.mul(input, torch.exp(log_s) + t)
        output = x0 + torch.mul(tmp, 1-binary_mask)
        # add Batch Normalization 
        return output # Determinant
    
    def reverse_forward(self, input, binary_mask):
        # FROM Z TO X
        pass


class Net(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def reverse_forward(self, input):
        pass