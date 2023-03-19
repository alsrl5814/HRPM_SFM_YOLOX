#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        
        self.conv_p3 = BaseConv(
            int(in_channels[0] * width), 1, 1, 1, act=act
        )
        
        self.conv_n3 = BaseConv(
            int(in_channels[1] * width), 1, 1, 1, act=act
        )
        
        self.conv_n4 = BaseConv(
            int(in_channels[2] * width), 1, 1, 1, act=act
        )
        
        self.downsample = Conv(
            1, 1, 3, 2, act=act
        )
        
        self.sigmoid = nn.Sigmoid()
        
        
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
    

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #DICONV_featuremap = Before_Module.DICONV
        #print(DICONV_featuremap.shape)
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        #print(x2.shape, x1.shape, x0.shape)
        
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        #print(f_out0.shape)
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        
        """pan_out0 = self.upsample(pan_out0)
        pan_out1 = self.upsample(pan_out1)
        pan_out2 = self.upsample(pan_out2)"""
        
        #print(pan_out0.shape, pan_out1.shape, pan_out2.shape)
        
        
        
        """vis_panout0 = torch.matmul(pan_out2,x2)
        print(vis_panout0.shape)
        print(vis_panout0[0,:,:,:].mean(axis=0).shape)
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        vis = cv2.resize(vis_panout0[0,:,:,:].mean(axis=0), dsize = (896,896), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis, "jet")
        plt.colorbar()
        #plt.clim(0, 0.4)
        plt.show()"""
        """import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        for i in range(0, vis_panout0.shape[1]):
            vis = cv2.resize(vis_panout0[0,i,:,:], dsize = (640,640), interpolation = cv2.INTER_CUBIC)
            name_num =  "./x2/C_" + str(i) +".jpg"
            
            plt.imshow(vis, 'jet')
            plt.savefig(name_num)
            #plt.colorbar()
            #plt.show()"""
        
        ### after module creation
        
        
        pan_out2_p3 = torch.mean(pan_out2, dim=1, keepdim = True)
        pan_out1_n3 = torch.mean(pan_out1, dim=1, keepdim = True)
        pan_out0_n4 = torch.mean(pan_out0, dim=1, keepdim = True)
        
        pan_out2_p3 = self.sigmoid(pan_out2_p3)
        pan_out1_n3 = self.sigmoid(pan_out1_n3)
        pan_out0_n4 = self.sigmoid(pan_out0_n4)
        #print(pan_out2_p3.shape, pan_out1_n3.shape, pan_out0_n4.shape)
        
        pan_out2 = pan_out2 * self.upsample(pan_out1_n3)
        pan_out1 = pan_out1 * self.upsample(pan_out0_n4)
        pan_out0 = pan_out0 * self.downsample(self.downsample(pan_out2_p3))
    
        """vis_panout0 = self.upsample(pan_out1_n3)
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        print(vis_panout0[0,:,:,:].shape, "************")
        print(vis_panout0[0,:,:,:])
        vis = cv2.resize(vis_panout0[0,0,:,:], dsize = (896,896), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis, "jet")
        plt.colorbar()
        #plt.clim(0, 0.4)
        plt.show()"""
        
        """vis_panout0 = pan_out0
        print(vis_panout0.shape)
        print(vis_panout0[0,:,:,:].mean(axis=0).shape)
        vis_panout0 = vis_panout0.detach().cpu().numpy()
        import matplotlib   
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        vis = cv2.resize(vis_panout0[0,:,:,:].mean(axis=0), dsize = (896,896), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis, "jet")
        plt.colorbar()
        #plt.clim(0, 0.4)
        plt.show()"""
        
        """vis_panout0 = pan_out2
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
        vis = cv2.resize(vis_panout0[0,:,:60,:].mean(axis=0), dsize = (320,240), interpolation = cv2.INTER_CUBIC)
        plt.imshow(vis, "jet")
        plt.colorbar()
        #plt.clim(0, 0.4)
        plt.show()"""
        outputs = (pan_out2, pan_out1, pan_out0)
        #print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
        return outputs
