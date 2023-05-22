import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return torch.reshape(x, self.shape)
    

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)
    
class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.squeeze(x, self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)
    
class ProcessRecurrentOutput(nn.Module):
    def __init__(self, output_id, hidden_dim):
        super(ProcessRecurrentOutput, self).__init__()
        self.output_id = output_id
        self.hidden_dim = hidden_dim
    def forward(self, x):
        import ipdb; ipdb.set_trace()
        x = x[self.output_id]
        x = x.contiguous().view(-1, self.hidden_dim)
        return x