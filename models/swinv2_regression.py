# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from models.swin_transformer_v2 import SwinTransformerV2


class SwinV2Regression(nn.Module):
    
    '''
        taken from https://github.com/SwinTransformer/MIM-Depth-Estimation/blob/main/models/model.py#L17
        adapt the depth estimation to 
    '''
    def __init__(self, args=None):
        super().__init__()
        # self.max_depth = args.max_depth
        
        embed_dim = 128
        num_heads = [4, 8, 16, 32]
        self.depths = [2, 2, 18, 2]
        self.num_filters = [32, 32, 32]
        self.deconv_kernels = [2, 2, 2]
        self.window_size = [30, 30, 30, 15]
        self.pretrain_window_size = [12, 12, 12, 6]
        self.use_shift = [True, True, False, False]
        self.drop_path_rate = 0.3
        self.use_checkpoint = False
        self.pretrained = "models/swinv2_base_1k.pth"
        self.out_indices = [0, 1, 2, 3]

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=self.depths,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrain_window_size=self.pretrain_window_size,
            drop_path_rate=self.drop_path_rate,
            use_checkpoint=self.use_checkpoint,
            use_shift=self.use_shift,
            out_indices = self.out_indices
        )

        self.encoder.init_weights(pretrained=self.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim
            
        self.decoder = Decoder(channels_in, channels_out)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
        self.readout = nn.Sequential(
            nn.Sigmoid())

    def forward(self, x):                
        conv_feats = self.encoder(x)
        out = self.decoder([conv_feats[3]])
        out = self.last_layer_depth(out)
        out = self.readout(out)
        # out_depth = torch.sigmoid(out_depth) * self.max_depth
        # x is a list, while the out is the final predicted results
        return out, conv_feats


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_deconv = 3
        self.deconv = self.num_deconv
        self.num_filters = [32, 32, 32]
        self.deconv_kernels = [2, 2, 2]
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv,
            self.num_filters,
            self.deconv_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=self.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        # import pdb; pdb.set_trace()
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

