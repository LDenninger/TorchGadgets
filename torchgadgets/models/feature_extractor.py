import torch.nn as nn
import torchvision.models as tv_models



RESNET_FEATURE_DIM = {
    18:{
        0: 512,
        1: 25088,
        2: 50176
    },
    50:{
        0: 2048,
        1: 100352,
        2: 200704
    }
}

CONVNEXT_FEATURE_DIM = {
    "tiny" : {
        0: 1000,
        1: 768,
        2: 37632,
        3: 75264,
    }
}

class ResNet(nn.Module):
    """
        Class that bundles the ResNet architecture in different configurations.

        Parameters:
            depth (int): Size of the ResNet architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size=50, layer=1, weights='DEFAULT'):
        super(ResNet, self).__init__()
        assert size in [18, 34, 50, 101, 152], 'Please provide a valid size from: 18, 34, 50, 101, 152'
        resnet_dict = {
            18: tv_models.resnet18,
            34: tv_models.resnet34,
            50: tv_models.resnet50,
            101: tv_models.resnet101,
            152: tv_models.resnet152
        }
        resnet = resnet_dict[size](weights=weights)
        # remove last FC layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-(1+layer)])

    def forward(self, input_):
        out = self.resnet(input_)
        out = out.view(out.shape[0], -1)
        return out
    

    
class ConvNeXt(nn.Module):
    """
        Class that bundles the ConvNeXt architecture in different configurations.

        Parameters:
            model (str): Size of the ConvNeXt architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size='tiny', layer=1, weights='DEFAULT'):
        super(ResNet, self).__init__()
        assert size in ['tiny','small', 'base', 'large'], 'Please provide a valid size from: tiny, small, base, large'
        convnext_dict = {
            'tiny': tv_models.convnext_tiny,
            'small': tv_models.convnext_small,
            'base': tv_models.convnext_base,
            'large': tv_models.convnext_large
        }
        convnext = convnext_dict[size][0](weights=weights)

        # remove last FC layer
        self.convnext = nn.Sequential(*list(convnext.children())[:-(1+layer)])

    def forward(self, input_):
        out = self.convnext(input_)
        out = out.view(out.shape[0], -1)
        return out
    
class VGG(nn.Module):
    """
        Class that bundles the ResNet architecture in different configurations.

        Parameters:
            depth (int): Size of the ResNet architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size=11, batch_norm=False, layer=1, weights='DEFAULT'):
        super(ResNet, self).__init__()
        assert size in [11, 13, 16, 19], 'Please provide a valid size from: 11, 13, 16, 19'
        if not batch_norm:
            vgg_dict = {
                11: tv_models.vgg11,
                13: tv_models.vgg13,
                16: tv_models.vgg16,
                19: tv_models.vgg19,
            }
        else:
            vgg_dict = {
                11: tv_models.vgg11_bn,
                13: tv_models.vgg13_bn,
                16: tv_models.vgg16_bn,
                19: tv_models.vgg19_bn,
            }
        vgg = vgg_dict[size](weights=weights)
        # remove last FC layer
        self.vgg = nn.Sequential(*list(vgg.children())[:-(1+layer)])

    def forward(self, input_):
        out = self.vgg(input_)
        out = out.view(out.shape[0], -1)
        return out

class MobileNetV3(nn.Module):
    """
        Class that bundles the ResNet architecture in different configurations.

        Parameters:
            depth (int): Size of the ResNet architecture.
            layer (int): Number of layers to remove from the back. 
                            Default is 1, meaning only the last fully connected layer is removed.
            weights (str): Which pretrained model weights to use.
                                Default is using pretrained weights from the ImageNet-1k dataset.
    
    """

    def __init__(self, size='small', layer=1, weights='DEFAULT'):
        super(ResNet, self).__init__()
        assert size in [11, 13, 16, 19], 'Please provide a valid size from: 11, 13, 16, 19'
        mobilenet_dict = {
            'small': tv_models.mobilenet_v3_small,
            'large': tv_models.mobilenet_v3_large,
        }
 
        mobilenet = mobilenet_dict[size](weights=weights)
        # remove last FC layer
        self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-(1+layer)])

    def forward(self, input_):
        out = self.mobilenet(input_)
        out = out.view(out.shape[0], -1)
        return out
    