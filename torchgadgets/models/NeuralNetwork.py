import torch
import torch.nn as nn

from .feature_extractor import *
from .ConvLSTM import *
from .util_modules import *
from .utils import *

class NeuralNetwork(nn.Module):
    
    def __init__(self, 
                    layers: list,):


        super(NeuralNetwork, self).__init__()
        self.backbone = None

        modules = build_model(layers)
        self.model = nn.Sequential(*modules)

    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def freeze_backbone_model(self):
        assert self.backbone is not None, 'No backbone model defined'
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone_model(self):
        assert self.backbone is not None, 'No backbone model defined'
        for param in self.backbone.parameters():
            param.requires_grad = True



