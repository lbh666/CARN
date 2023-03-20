from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import sys
def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer



def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True,
               groups=1):
    """
    Re-write convolution layer for adaptive `padding`.
    """

    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
            int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    bias=bias,
                    groups=groups)

def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class CAESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, n_feats):
        super(CAESA, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            activation('lrelu', neg_slope=0.05),
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        return x * self.se(x)

class CARN(nn.Module):
    def __init__(self,
                in_channels=3,
                out_channels=3,
                feature_channels=64,
                upscale=4,):
        super(CARN, self).__init__()
        

        self.conv_1 = conv_layer(in_channels,
                                feature_channels,
                                kernel_size=3)
        
        self.block_1 = deployRLFB_rrrb(feature_channels)
        self.block_2 = deployRLFB_rrrb(feature_channels)
        self.block_3 = deployRLFB_rrrb(feature_channels)
        self.block_4 = deployRLFB_rrrb(feature_channels)


        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)
        
    def forward(self, x):
        out_feature = self.conv_1(x)

        y = self.block_1(out_feature)
        y = self.block_2(y)
        y = self.block_3(y)
        y = self.block_4(y)

        out_low_resolution = self.conv_2(y) + out_feature
        output = self.upsampler(out_low_resolution)

        return output

class deployRLFB_rrrb(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,):
        super(deployRLFB_rrrb, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)


        self.esa = CAESA(out_channels)


        self.act = activation('lrelu', neg_slope=0.05)
        


    def forward(self, x):


        x = (self.c1_r(x))
        x = self.act(x)

        x = (self.c2_r(x))
        x = self.act(x)
        x = (self.c3_r(x))
        x = self.act(x)
        x = self.esa(x)
        
        return x


    