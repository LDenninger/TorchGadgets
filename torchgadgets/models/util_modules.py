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
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)
    
class ProcessRecurrentOutput(nn.Module):
    """
        Module to process the raw output of a recurrent module,
        such that it can be further used withing the neural network.
    
    """
    def __init__(self, output_id, hidden_dim, sequence=None, batch_first=False):
        super(ProcessRecurrentOutput, self).__init__()
        self.output_id = output_id
        self.hidden_dim = hidden_dim
        self.sequence = sequence
        self.batch_first = batch_first
        if isinstance(self.hidden_dim, int):
            self.hidden_dim = [self.hidden_dim]
    def forward(self, x):
        x = x[self.output_id]
        if self.sequence is not None:
            if self.batch_first:
                x = x[:,self.sequence,...]
            else:
                x = x[self.sequence,...]
        x = x.contiguous().view(-1, *self.hidden_dim)
        
        return x
    
class RecurrentCellWrapper(nn.Module):
    """
        A wrapper class for recurrent cells that implements sequence-wise computation
        for each cell.
    """

    def __init__(self, cell: nn.Module, return_seq_output=False, batch_first=False):
        super(RecurrentCellWrapper, self).__init__()
        self.cell = cell
        self.batch_first = batch_first
        self.return_seq_output = return_seq_output

    def forward(self, x, hx=None):
        """
            Forward through a single reccurent cell.

            Arguments:
                x: input sequence 
                    Shape: [batch_size, sequence_size, input_size], if batched else [sequence_size, input_size]
                hx: Initial hidden state
        """
        batched = len(x.shape)==3
        seq_len = x.shape[1] if batched else x.shape[0]
        if batched and self.batch_first:
            x = x.transpose(0, 1)

        seq_result = []
        
        for sid in range(seq_len):
            result = self.cell(x[sid], hx)
            seq_result.append(result)

        if self.return_seq_output:
            seq_result = torch.stack(seq_result, dim=0)
            if self.batch_first:
                seq_result = seq_result.transpose(0, 1)
            return seq_result

        return self.cell(x, hx)
    