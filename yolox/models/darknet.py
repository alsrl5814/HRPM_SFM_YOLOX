#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from pickletools import uint8
from re import X
from pexpect import EOF
import torch
from torch import dtype, nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck, Before_Module


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        #self.stem = Focus(3, 6, ksize=3, act=act)
        #self.stem2 = Focus2(6, base_channels, ksize=1, act=act)
        #self.thresh_relu_BE = BaseConv(6, 1, 1, stride=1, act="relu")
        #self.thresh_relu_EO = BaseConv(6, 1, 1, stride=1, act="relu")
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding= 0)
        
        
        #Before Module
        self.before_module = Before_Module(base_channels, base_channels, ksize=3, act=act)
        
        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        
        x = self.before_module(x)
        """mean_vis = x[0,:,:,:].mean(axis=0)
        
        max_mean_vis = mean_vis[:,:].max()
        min_mean_vis = mean_vis[:,:].min()
        mean_mean_vis = mean_vis[:,:].mean()
        max_mean_mid_vis = (max_mean_vis + mean_mean_vis) / 2
        min_mean_mid_vis = (min_mean_vis + mean_mean_vis) / 2
        
        #print(max_mean_vis,min_mean_vis,mean_mean_vis,max_mean_mid_vis,min_mean_mid_vis)
        #print(self.thresh_relu_BE(x).dtype)
        if self.thresh_relu_BE(x).is_cuda:
            BE = torch.where(self.thresh_relu_BE(x) > min_mean_mid_vis, self.thresh_relu_BE(x), torch.Tensor([0.]).to(device='cuda:0').type(self.thresh_relu_BE(x).dtype))
        else:   
            BE = torch.where(self.thresh_relu_BE(x) > min_mean_mid_vis, self.thresh_relu_BE(x), torch.Tensor([0.]).cpu().float())
        
        if self.thresh_relu_EO(x).is_cuda:
            EO = torch.where(self.thresh_relu_EO(x) > max_mean_mid_vis, self.thresh_relu_EO(x), torch.Tensor([0.]).to(device='cuda:0').type(self.thresh_relu_EO(x).dtype))
        else:   
            EO = torch.where(self.thresh_relu_EO(x) > max_mean_mid_vis, self.thresh_relu_EO(x), torch.Tensor([0.]).cpu().float())
        
        edge_feature = EO-BE
        x = x * (edge_feature)
        #print(x.shape, "broadcasting mul after")
        
        x = self.stem2(x)
        #print(x.shape, "stem2")
        x = x * self.maxpool(edge_feature)
        #print(x.shape, "stem2 + maxpool")"""
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        """vis_panout0 = x
        print(vis_panout0.shape)
        print(vis_panout0[0,:,:,:].mean(axis=0).shape)
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        #plt.figure(figsize = (37.041666666667,37.041666666667))
        vis = cv2.resize(vis_panout0[0,:,:,:].mean(axis=0), dsize = (896,896), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis, "jet")
        plt.colorbar()
        #plt.clim(1.0, 2.5)
        plt.show()"""
        
        
        """for i in range(0, vis_panout0.shape[1]):
            vis = cv2.resize(vis_panout0[0,i,:,:], dsize = (640,640), interpolation = cv2.INTER_CUBIC)
            name_num =  "./stem_relu/C_" + str(i) +".jpg"
            
            plt.imshow(vis, 'jet')
            plt.savefig(name_num)
            #plt.colorbar()
            #plt.show()"""
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
