#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class DilateConv(nn.Module):
    "A Conv2d -> Batchnorm -> silu/leaky relu block"

    def __init__(
        self, in_channels, out_channels, ksize, stride, dilate = 2, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2 + 1 
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=dilate,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels
        

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)
        #self.conv = BaseConv(in_channels, out_channels, ksize, stride, act=act)
        #self.max_pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        
        # shape of x (b,3,w,h) -> y(b,12,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        """vis_panout0 = x
        #vis_panout0 = self.max_pool(vis_panout0)
        #vis_panout0 -= vis_panout0.min()
        #vis_panout0 /= vis_panout0.max()
        #print(vis_panout0.shape)
        #print(vis_panout0[0,:,:,:].mean(axis=0).shape)
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        #vis = cv2.resize(vis_panout0[0,:,:,:].mean(axis=0), dsize = (640,640), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis_panout0[0,:3,:,:])
        plt.colorbar()
        #plt.clim(0, 0.4)
        plt.show()"""
        x = self.conv(x)
        #print(x.shape)
        return x
class Focus2(nn.Module):
    "Focus width and height information into channel space."

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv_before = BaseConv(in_channels, 3, ksize, stride, act=act)
        self.conv_after = BaseConv(12, out_channels, ksize, stride, act=act)
    def forward(self, x):
        x = self.conv_before(x)
        # shape of x (b,12,w,h) ->(b,3,w,h) -> y(b,12,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        x = self.conv_after(x)
        
        return x
    
class Before_Module(nn.Module):
    "Focus width and height information into channel space."

    def __init__(self, in_channels, out_channels, ksize=1, stride=2, act="silu"):
        super().__init__()
        #self.conv_before = DilateConv(in_channels, out_channels, ksize, stride=3, dilate=3, act=act)
        self.conv_Dilate1 = DilateConv(in_channels, in_channels // 2, ksize, stride=1, dilate=2, act=act)
        self.conv_Dilate2 = DilateConv(in_channels, in_channels // 2, ksize, stride=2, dilate=2, act=act)
        self.conv_Base = BaseConv(in_channels, in_channels // 2, ksize=3, stride=2, act=act)
        self.conv_Base2 = BaseConv(in_channels, in_channels // 2, ksize, stride=1, act=act)
        self.max_pool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        
        #print(x.shape)
        Diconv = self.conv_Dilate2(x) ##480 480 12
        #print(Diconv.shape)
        maxconv = self.max_pool(x) ##480 480 24
        #print(maxconv.shape)
        # context feature + obj feature
        maxconv = torch.max(maxconv[:, ::2 , : , : ], maxconv[: , 1::2, : ,: ])
        #print(maxconv.shape)
        """vis_panout0 = maxconv
        vis_panout0 -= vis_panout0.min()
        vis_panout0 /= vis_panout0.max()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        for i in range(0, vis_panout0.shape[1]):
            vis = cv2.resize(vis_panout0[0,i,:240,:], dsize = (320,240), interpolation = cv2.INTER_CUBIC)
            name_num =  "./maxcompression/C_" + str(i) +".jpg"
            
            plt.imshow(vis, 'jet')
            plt.savefig(name_num)
            #plt.colorbar()
            #plt.show()"""
        
        # max compression을 통해 object edge feature 다양한게 학습된 filter 정보를 하나로 합치면서 연산량 경량화
        Baseconv = self.conv_Base(x) ## 480 480 24
        #print(Baseconv.shape)
        #print(Baseconv.shape, " ##################################")
        Baseconv2 = torch.cat((Baseconv,maxconv), dim=1)
        #print(Baseconv2.shape)
        x = torch.cat((Diconv, maxconv), dim=1) ##480 480 24
        #print(x.shape)
        x = self.conv_Base2(x) ##480 480 12
        #print(x.shape)
        Diconv2 = self.conv_Dilate1(Baseconv2) ## 480 480 12
        """vis_panout0 = Diconv2
        vis_panout0 -= vis_panout0.min()
        vis_panout0 /= vis_panout0.max()
        print(vis_panout0.shape)
        print(vis_panout0[0,:,:,:].mean(axis=0).shape)
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        vis = cv2.resize(vis_panout0[0,:,:240,:].mean(axis=0), dsize = (320,240), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis, "jet")
        plt.colorbar()
        #plt.clim(0, 0.4)
        plt.show()"""
        x = torch.cat((x, Diconv2), dim=1) ##480 480 32
        #print(x.shape)
        return x
