import torch
import torch.nn as nn

from .feature_extractor import *
from .ConvLSTM import *
from .util_modules import *



def build_model(layer_config):
    """
        Function to build a PyTorch model from the layer config.
    
    """

    layers = nn.ModuleList()
    for (i, layer) in enumerate(layer_config):

        ##-- Feature Extractor Models --##
        if layer["type"] == "ResNet":
            layers.append(ResNet(size=layer['size'], layer=layer['remove_layer'], weights=layer['weights']))
        elif layer["type"] == "ConvNext":
            layers.append(ConvNeXt(size=layer['size'], layer=layer['remove_layer'], weights=layer['weights']))
        elif layer["type"] == "VGG":
            layers.append(VGG(size=layer['size'], layer=layer['remove_layer'], weights=layer['weights']))
        elif layer["type"] == "MobileNetV3":
            layers.append(MobileNetV3(size=layer['size'], layer=layer['remove_layer'], weights=layer['weights']))
        elif layer["type"] == "ViT":
            layers.append(VisualTransformer(size=layer['size'], layer=layer['remove_layer'], weights=layer['weights']))

        ##-- MLP Layers --##
        elif layer['type'] == 'linear':
            layers.append(nn.Linear(in_features=layer['in_features'], out_features=layer['out_features']))
        elif layer['type'] == 'dropout':
            layers.append(nn.Dropout(layer['prob']))
        
        
        ##-- CNN Layers --##
        elif layer["type"] == "conv2d":
            layers.append(nn.Conv2d(layer["in_channels"], layer["out_channels"], layer["kernel_size"], layer["stride"], layer["padding"]))
        elif layer["type"] == "conv3d":
            layers.append(nn.Conv3d(layer["in_channels"], layer["out_channels"], layer["kernel_size"], layer["stride"], layer["padding"]))
        elif layer["type"] == "conv1d":
            layers.append(nn.Conv1d(layer["in_channels"], layer["out_channels"], layer["kernel_size"], layer["stride"], layer["padding"]))
        elif layer["type"] == "transConv2d":
            layers.append(nn.ConvTranspose2d(layer["in_channels"], layer["out_channels"], layer["kernel_size"], layer["stride"], layer["padding"], output_padding=layer["output_padding"]))
        #- Pooling -#
        elif layer["type"] == "maxpool2d":
            layers.append(nn.MaxPool2d(layer["kernel_size"], layer['stride']))
        elif layer["type"] == "avgpool2d":
            layers.append(nn.AvgPool2d(layer["kernel_size"], layer['stride']))
        elif layer["type"] == "ada_maxpool2d":
            layers.append(nn.AdaptiveMaxPool2d(layer["output_size"]))
        elif layer["type"] == "ada_avgpool2d":
            layers.append(nn.AdaptiveAvgPool2d(layer["output_size"]))



        ##-- Recurrent Models --##
        elif layer["type"] == "RNN":
            layers.append(nn.RNN(input_size=layer["input_size"], 
                                    hidden_size=layer["hidden_size"], 
                                        dropout=layer['dropout'],
                                            bidirectional=layer['bidirectional'],
                                                nonlinearity=layer['activation'], 
                                                    num_layers=layer["num_layers"], 
                                                        batch_first=layer['batch_first']))
            layers.append(ProcessRecurrentOutput(layer['output_id'], layer['hidden_size'], None if layer['sequence_out']=='all' else layer['sequence_out'], layer['batch_first']))
        elif layer["type"] == "LSTM":
            layers.append(nn.LSTM(input_size=layer["input_size"], 
                                    hidden_size=layer["hidden_size"],
                                        num_layers=layer["num_layers"],
                                            dropout=layer['dropout'],
                                                bidirectional=layer['bidirectional'],
                                                    batch_first=layer['batch_first']))
            layers.append(ProcessRecurrentOutput(layer['output_id'], layer['hidden_size'], None if layer['sequence_out']=='all' else layer['sequence_out'], layer['batch_first']))

        elif layer["type"] == "GRU":
            layers.append(nn.GRU(input_size=layer["input_size"], 
                                    hidden_size=layer["hidden_size"], 
                                        num_layers=layer["num_layers"],
                                            dropout=layer['dropout'],
                                                bidirectional=layer['bidirectional']))
            layers.append(ProcessRecurrentOutput(layer['output_id'], layer['hidden_size'], None if layer['sequence_out']=='all' else layer['sequence_out'], layer['batch_first']))
            
        elif layer["type"] == "ConvLSTM":
            layers.append(ConvLSTM( layers = layer['layers'],
                                        input_dims=layer["input_size"], 
                                            batch_first = layer['batch_first']))
            layers.append(ProcessRecurrentOutput(layer['output_id'], (layer['layers'][-1]['hidden_size'], layer['input_size'][1], layer['input_size'][2]), None if layer['sequence_out']=='all' else layer['sequence_out'], layer['batch_first']))
        
        ##-- Recurrent Cells --##
        elif layer["type"] == "RNNCell":
            layers.append(RecurrentCellWrapper( nn.RNNCell(input_size=layer["input_size"], 
                                                            hidden_size=layer["hidden_size"], 
                                                                nonlinearity=layer['activation']),
                                                batch_first=True))
        elif layer["type"] == "LSTMCell":
            layers.append(RecurrentCellWrapper( nn.LSTMCell(input_size=layer["input_size"], 
                                                                hidden_size=layer["hidden_size"]),
                                                batch_first=True))
        elif layer["type"] == "GRUCell":
            layers.append(RecurrentCellWrapper( nn.GRUCell(input_size=layer["input_size"], 
                                                            hidden_size=layer["hidden_size"]),
                                                batch_first=True))
            
        elif layer["type"] == "ConvLSTMCell":
            layers.append(RecurrentCellWrapper( ConvLSTMCell(input_size=layer["input_size"], 
                                                    hidden_size=layer["hidden_size"], 
                                                        kernel_size=layer["kernel_size"],
                                                            stride=layer["stride"],
                                                                batchnorm=layer['batchnorm']),
                                                batch_first=True))
        
        ##-- Normalization Layers --##
        elif layer['type'] =='batchnorm1d':
            layers.append(nn.BatchNorm1d(layer['num_features'], layer['eps'], layer['momentum']))

        elif layer['type'] =='batchnorm2d':
            layers.append(nn.BatchNorm2d(layer['num_features'], layer['eps'], layer['momentum']))

        ##-- Input Manipulation --##
        elif layer['type'] == 'flatten':
            layers.append(nn.Flatten(start_dim=layer['start_dim'], end_dim=layer['end_dim']))
        elif layer['type'] == 'unsqueeze':
            layers.append(Unsqueeze(dim=layer['dim']))
        elif layer['type'] =='squeeze':
            if layer['dim'] == 'all':
                layers.append(Squeeze())
            else:
                layers.append(Squeeze(dim=layer['dim']))
        elif layer['type'] =='permute':
            layers.append(Permute(layer['dim']))
        elif layer['type'] =='reshape':
            layers.append(Reshape(layer['shape']))
        elif layer['type'] == 'unflatten':
            layers.append(nn.Unflatten())
        elif layer['type'] == 'repeat':
            layers.append(Repeat(dim=layer['dim'], num_repeat=layer['num_repeat'], unsqueeze=layer['unsqueeze']))
        ##-- Activation Functions --##
        elif layer["type"] == "relu":
            layers.append(nn.ReLU())
        elif layer['type'] == 'leaky_relu':
            layers.append(nn.LeakyReLU(layer['negative_slope']))
        elif layer['type'] == 'gelu':
            layers.append(nn.GELU())
        elif layer['type'] == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif layer["type"] == "softmax":
            layers.append(nn.Softmax(dim=-1))

        ##-- Custom Module --##
        elif layer["type"] == "custom":
            layers.append(layer['module'])
    
    return layers