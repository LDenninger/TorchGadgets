import torch
import torch.nn as nn

class ConvolutionalNN(nn.Module):
    
    def __init__(self, 
                    layers: list,):
        """
            Convolutional Neural Network. The layers are fully configurable by providing a simple config for each layer

            Parameters:
                layers (list): list of layers to be configured
                    Format:
                            [
                                {
                                    "type": "conv2d",
                                    "in_channels": 1,
                                    "out_channels": 1,
                                    "kernel_size": (3,3),
                                    "stride": (1,1),
                                }, 
                                {
                                    "type": "relu"
                                },
                                {
                                    "type": "maxpool2d",
                                    "kernel_size": (2,2)
                                },
                                {
                                    "type": "avgpool2d",
                                    "kernel_size": (2,2)
                                },
                                {
                                    "type": "linear",
                                    "in_features": 512,
                                    "out_features": 10
                                }
        """

        super(ConvolutionalNN, self).__init__()

        self.build_model(layers)


    def build_model(self, layer_config):
        layers = nn.ModuleList()
        for (i, layer) in enumerate(layer_config):
            if layer["type"] == "conv2d":
                layers.append(nn.Conv2d(layer["in_channels"], layer["out_channels"], layer["kernel_size"], layer["stride"]))
            elif layer["type"] == "relu":
                layers.append(nn.ReLU())
            elif layer["type"] == "softmax":
                layers.append(nn.Softmax(dim=-1))
            elif layer["type"] == "maxpool2d":
                layers.append(nn.MaxPool2d(layer["kernel_size"], layer['stride']))
            elif layer["type"] == "avgpool2d":
                layers.append(nn.AvgPool2d(layer["kernel_size"], layer['stride']))
            # Add a linear layer with a given activation function
            elif layer['type'] == 'linear':
                layers.append(nn.Linear(in_features=layer['in_features'], out_features=layer['out_features']))
            elif layer['type'] =='batchnorm2d':
                layers.append(nn.BatchNorm2d(layer['num_features'], layer['eps'], layer['momentum']))
            elif layer['type'] == 'dropout':
                layers.append(nn.Dropout(layer['prob']))
            elif layer['type'] == 'flatten':
                layers.append(nn.Flatten())
            
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.model(x)
        return x
        