import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    """
        Convolutional LSTM Cell.

        Implementation of a ConvLSTM cell as proposed by Shi et al.(2015) in "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nwocasting".
            ---> https://arxiv.org/pdf/1506.04214.pdf
        
        The output of the ConvLSTm cell is computed in vectorized form according to:
            A_t = Convolution((input, hidden_state))  (Concatenated Gate Matrix)
            i_t = sigmoid(A_t[:,0:hidden_size])
            f_t = sigmoid(A_t[:,hidden_size:2*hidden_size])
            o_t = sigmoid(A_t[:,2*hidden_size:3*hidden_size])
            g_t = tanh(A_t[:,3*hidden_size:4*hidden_size])
            c_t = f_t * c_{t-1} + i_t * g_t
            h_t = o_t * tanh(c_t)
    """

    def __init__(self,  in_channels: int,
                            input_size: tuple[int, int], 
                                hidden_size: int,
                                    kernel_size: tuple[int, int],
                                        stride: int = 1,
                                            bias: bool = True,
                                                batchnorm: bool = False,
                                                    device: str = 'cuda') -> None:
        """
            Initialize a ConvLSTM cell.

            Parameters:
                in_channels (int): Number of input channels
                input_size tuple(int, int): Size of the input image (height, width)
                hidden_size (int): Channels of the hidden state
                kernel_size tuple(int, int): Kernel size used in the convolution
                stride (int, optional): Stride of the convolutional kernel
                bias (bool, optional): Whether to add a bias term to the convolution. Default: True
                batchnorm (bool, optional): Whether to use batch normalization to the cell. Default: False
                device (str, optional): Device to run the model. Default: 'cuda'
        """
        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.stride = stride
        self.bias = bias
        self.batchnorm = batchnorm
        self.device = device

        self.padding = int((self.kernel_size[0] - 1) // 2)

        self.Conv = nn.Conv2d(self.in_channels+ self.hidden_size, 4*self.hidden_size, kernel_size=self.kernel_size, stride=self.stride, bias=self.bias, padding=self.padding)

        if self.batchnorm:
            self.conv_bn = nn.BatchNorm2d(4*self.hidden_size)
            self.out_bn = nn.BatchNorm1d(self.hidden_size)


    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Forward function of the ConvLSTM cell. 
            The function computes a LSTM cell in vectorized form given a 3D input.
            
            Arguments:
                input: Input image, [batch_size, in_channels, height, width]
                hidden_state: Hidden state of form (h_t, c_t), 
                 -> Shape: ([batch_size, hidden_size, height, width], [batch_size, hidden_size, height, width])
                 -> Hidden state must be provided. If this is the first call in a sequence,
                    one can usethe init_hidden() function priorly to initialize the hidden state
            
            Output: Hidden State (h_t, c_t)
        """
        
        hidden, c_t = hidden_state

        # Concatenate the input and the hidden state in the channel dimension
        input_concat = torch.cat((input, hidden), dim=1)
        # Concatenated Gate Matrix
        A_t = self.Conv(input_concat)
        # Introduce batch normalization at the input
        if self.batchnorm:
            A_t = self.conv_bn(A_t)
        # Simply apply formulas to the vectorized version
        i_t = torch.sigmoid(A_t[:,0:self.hidden_size])
        f_t = torch.sigmoid(A_t[:,self.hidden_size:2*self.hidden_size])
        o_t = torch.sigmoid(A_t[:,2*self.hidden_size:3*self.hidden_size])
        g_t = torch.tanh(A_t[:,3*self.hidden_size:4*self.hidden_size])
        c_t = f_t * c_t + i_t * g_t
        # Apply batch normalization before applying the activation
        if self.batchnorm:
            h_t = o_t * torch.tanh(self.out_bn(c_t))
        else:
            h_t = o_t * torch.tanh(c_t)


        return h_t, c_t

    def init_hidden(self,batch_size):
        """
            Initialize the hidden state of the ConvLSTM cell with Zeros.

            Output: Zeros of shape: 
                ([batch_size, hidden_size, height, width], [batch_size, hidden_size, height, width])
            
        
        """
        return (torch.zeros(batch_size,self.hidden_size,self.input_size[0],self.input_size[1]).to(self.device),torch.zeros(batch_size,self.hidden_size,self.input_size[0],self.input_size[1]).to(self.device))

    
class ConvLSTM(nn.Module):

    """
        Convolutional LSTM consisting of multiple layers of ConvLSTMCell modules.
    
    """

    def __init__(self,  layers: list,
                            input_dims: tuple[int, int, int], 
                                    bias: bool = True,
                                        batch_first: bool = False,
                                            device: str = 'cuda') -> None:
        super(ConvLSTM, self).__init__()

        self.in_channels = input_dims[0]
        self.input_size = (input_dims[1], input_dims[2])
        self.bias = bias
        self.batch_first = batch_first
        self.num_layers = len(layers)
        self.device = device

        cells = []

        for i, layer in enumerate(layers):
                cells.append(ConvLSTMCell( in_channels = self.in_channels if i==0 else layers[i-1]['hidden_size'],
                                                input_size=self.input_size, 
                                                    hidden_size=layer['hidden_size'],
                                                        kernel_size=layer['kernel_size'], 
                                                            stride=layer['stride'],
                                                                batchnorm=layer['batchnorm'],
                                                                    bias=self.bias,
                                                                        device=self.device))
        
        self.model = nn.ModuleList(cells)

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Required input shape: (seq_len, batch_size, in_channels, height, width)
        if self.batch_first:
            input = input.permute(1, 0, 2, 3, 4)

        # If no initial hidden state is provided, initialize it with zeros
        if hidden_state is None:
            hidden_state = self.init_hidden(input.shape[1])

        hidden_output = []

        seq_len = input.shape[0]

        # Loop over the layers
        for i, layer in enumerate(self.model):

            hidden_c = hidden_state[i]
            inner_output = []
            
            # For each layer loop over the sequences
            for s_id in range(seq_len):
                hidden_c = layer(input[s_id], hidden_c)
                inner_output.append(hidden_c[0])
            # Gather output from each layer for each sequence        
            hidden_output.append(hidden_c)

            # Take the 
            input = torch.cat(inner_output, dim=0).view(input.shape[0], *inner_output[0].shape)
        # Output of last layer: [batch_size, hidden_size, height, width]
        output = hidden_c[0]
        return output, (hidden_output, input)

    def init_hidden(self,batch_size):
        init_states=[]#this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.model[i].init_hidden(batch_size))
        return init_states