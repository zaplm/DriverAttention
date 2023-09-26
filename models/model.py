import os
import numpy as np
from typing import Dict
import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
from models.MLNet import MLNet
import torch.nn.functional as F
from torchvision.models import resnet50
from models.nonLocal import NONLocalBlock2D
from models.models import ConvNextModel, ResNetModel, VGGModel, MobileNetV2, DenseModel
from models.MobileViT import mobile_vit_small
import cv2
from torch import Tensor


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ShortcutProjection(nn.Module):
    """
    ## Linear projections for shortcut connection
    This does the $W_s x$ projection described above.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `out_channels` is the number of channels in $\mathcal{F}(x, \{W_i\})$
        * `stride` is the stride length in the convolution operation for $F$.
        We do the same stride on the shortcut connection, to match the feature-map size.
        """
        super().__init__()

        # Convolution layer for linear projection $W_s x$
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # Paper suggests adding batch normalization after each convolution operation
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        # Convolution and batch normalization
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    """
    <a id="residual_block"></a>
    ## Residual Block
    This implements the residual block described in the paper.
    It has two $3 \times 3$ convolution layers.
    ![Residual Block](residual_block.svg)
    The first convolution layer maps from `in_channels` to `out_channels`,
    where the `out_channels` is higher than `in_channels` when we reduce the
    feature map size with a stride length greater than $1$.
    The second convolution layer maps from `out_channels` to `out_channels` and
    always has a stride length of 1.
    Both convolution layers are followed by batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `out_channels` is the number of output channels
        * `stride` is the stride length in the convolution operation.
        """
        super().__init__()

        # First $3 \times 3$ convolution layer, this maps to `out_channels`
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        # First activation function (ReLU)
        self.act1 = nn.ReLU()

        # Second $3 \times 3$ convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection should be a projection if the stride length is not $1$
        # of if the number of channels change
        if stride != 1 or in_channels != out_channels:
            # Projection $W_s x$
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            # Identity $x$
            self.shortcut = nn.Identity()

        # Second activation function (ReLU) (after adding the shortcut)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input of shape `[batch_size, in_channels, height, width]`
        """
        # Get the shortcut connection
        shortcut = self.shortcut(x)
        # First convolution and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution
        x = self.bn2(self.conv2(x))
        # Activation function after adding the shortcut
        return self.act2(x + shortcut)


class UncertaintyBlock(nn.Module):
    def __init__(self, img_input, p_input, n):
        super(UncertaintyBlock, self).__init__()
        self.rs1 = ResidualBlock(img_input, p_input, 1)
        list_block = []
        for i in range(n):
            list_block.append(ResidualBlock(p_input, p_input, 1))
        self.rs2 = nn.ModuleList(list_block)
        # 内容应该不同
        # self.rs3 = ResidualBlock(p_input * (n + 1), p_input, 1)
        self.nlb = NONLocalBlock2D(p_input * (n + 1))
        # self.seb = BasicBlock(p_input * (n + 1), p_input)
        list_block = []
        for i in range(n):
            list_block.append(ResidualBlock(p_input * (n + 2), p_input, 1))
        self.rs4 = nn.ModuleList(list_block)

    def forward(self, x, p):
        x = self.rs1(x)
        for index, m in enumerate(self.rs2):
            p[index] = m(p[index])
        c = x
        c = torch.concat([c, *p], dim=1)
        # c = self.nlb(c)
        # c = self.seb(c)
        # c = self.rs3(c)
        p = [torch.concat([ps, c], dim=1) for ps in p]
        for index, m in enumerate(self.rs4):
            p[index] = m(p[index])
        return p


class Model(nn.Module):
    def __init__(self, backbone, dim=32, input_dim=3):
        super(Model, self).__init__()
        self.n = len(os.listdir('./pseudo_labels'))
        list_block = []
        for i in range(self.n):
            list_block.append(Conv2dNormActivation(
                input_dim,
                dim,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,
                bias=True,
            ))
        self.first_conv = nn.ModuleList(list_block)
        # 不应该share weights
        self.mode = backbone

        if backbone == 'resnet':
            print('resnet backbone')
            self.backbone = ResNetModel(train_enc=True)
            self.ub1 = UncertaintyBlock(64, dim, self.n)
            self.ub2 = UncertaintyBlock(256, dim, self.n)
            self.ub3 = UncertaintyBlock(1024, dim, self.n)
        elif backbone == 'ConvNext':
            print('ConvNext backbone')
            self.backbone = ConvNextModel()
            self.ub1 = UncertaintyBlock(96, dim, self.n)
            self.ub2 = UncertaintyBlock(192, dim, self.n)
            self.ub3 = UncertaintyBlock(384, dim, self.n)
        elif backbone == 'mobileViT':
            print('mobileViT backbone')
            self.backbone = mobile_vit_small()
            weights_dict = torch.load('models/mobilevit_s.pt', map_location='cpu')
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "classifier" in k:
                    del weights_dict[k]
            self.backbone.load_state_dict(weights_dict, strict=False)
            self.ub1 = UncertaintyBlock(32, dim, self.n)
            self.ub2 = UncertaintyBlock(64, dim, self.n)
            self.ub3 = UncertaintyBlock(96, dim, self.n)
            self.ub4 = UncertaintyBlock(128, dim, self.n)
            self.ub5 = UncertaintyBlock(160, dim, self.n)
        elif backbone == 'vgg':
            print('vgg backbone')
            self.backbone = VGGModel(train_enc=True)
            self.ub1 = UncertaintyBlock(128, dim, self.n)
            self.ub2 = UncertaintyBlock(256, dim, self.n)
            self.ub3 = UncertaintyBlock(512, dim, self.n)
        elif backbone == 'mobilenet':
            print('mobilenet v2 backbone')
            self.backbone = MobileNetV2(train_enc=True)
            self.ub1 = UncertaintyBlock(16, dim, self.n)
            self.ub2 = UncertaintyBlock(24, dim, self.n)
            self.ub3 = UncertaintyBlock(96, dim, self.n)
        elif backbone == 'densenet':
            print('densenet backbone')
            self.backbone = DenseModel(train_enc=True)
            self.ub1 = UncertaintyBlock(96, dim, self.n)
            self.ub2 = UncertaintyBlock(192, dim, self.n)
            self.ub3 = UncertaintyBlock(1056, dim, self.n)
        list_block = []
        for i in range(self.n):
            list_block.append(nn.Conv2d(dim, 1, kernel_size=1))
        self.conv = nn.ModuleList(list_block)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample8x = nn.UpsamplingBilinear2d(scale_factor=8)

        self.downsample2x = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.apply(self.init_parameters)

        # self.nw_enc = nw.ResnetEncoder(18, False)
        # loaded_dict_enc = torch.load('models/encoder.pth')
        # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.nw_enc.state_dict()}
        # self.nw_enc.load_state_dict(filtered_dict_enc)
        # for param in self.nw_enc.parameters():
        #     param.requires_grad = False
        # self.nw_dec = nw.DepthDecoder(num_ch_enc=self.nw_enc.num_ch_enc, scales=range(4))
        # loaded_dict = torch.load('models/depth.pth')
        # self.nw_dec.load_state_dict(loaded_dict)
        # for param in self.nw_dec.parameters():
        #     param.requires_grad = False

        # self.activate = nn.Sigmoid()

    # @staticmethod
    # def init_parameters(m):
    #     if isinstance(m, nn.Conv2d):
    #         if m.weight is not None:
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out")
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
    #         if m.weight is not None:
    #             nn.init.ones_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, (nn.Linear,)):
    #         if m.weight is not None:
    #             nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     else:
    #         pass

    def forward(self, x, p=None):
        # features = self.nw_enc(x)
        # outputs = self.nw_dec(features)
        # disp = outputs[("disp", 0)]
        # x = x * (disp / disp.max())

        y, results = self.backbone(x)
        self.heatmap = self.process_output(y)
        if p is None:
            return y

        e = []
        for index, m in enumerate(self.first_conv):
            e.append(m(p[index]))
        if self.mode == 'resnet':
            results[0] = self.downsample2x(results[0])
            results[2] = self.upsample4x(results[2])
        elif self.mode == 'ConvNext':
            results[1] = self.upsample2x(results[1])
            results[2] = self.upsample4x(results[2])
        elif self.mode == 'mobileViT':
            results[0] = self.downsample2x(results[0])
            results[2] = self.upsample2x(results[2])
            results[3] = self.upsample4x(results[3])
            results[4] = self.upsample8x(results[4])
        elif self.mode == 'vgg':
            results[0] = self.downsample2x(results[0])
            results[2] = self.upsample4x(results[2])
        elif self.mode == 'mobilenet':
            results[0] = self.downsample2x(results[0])
            results[2] = self.upsample4x(results[2])
        elif self.mode == 'densenet':
            results[0] = self.downsample2x(results[0])
            results[2] = self.upsample4x(results[2])

        e = self.ub1(results[0], e)
        e = self.ub2(results[1], e)
        e = self.ub3(results[2], e)
        e = self.ub4(results[3], e)
        e = self.ub5(results[4], e)

        e = [self.upsample4x(ps) for ps in e]
        for index, m in enumerate(self.conv):
            e[index] = m(e[index])
        e = [torch.sigmoid(ps) for ps in e]
        # e = [torch.relu(ps) for ps in e]
        # p = [ps.split(1, dim=1) for ps in p]
        # e = [ps[0] for ps in p]
        # s = [ps[1] for ps in p]

        return y, e

    @staticmethod
    def process_output(outputs):
        # Parse the output of the model
        heatmap = torch.squeeze(outputs[-1]).data.cpu().numpy()

        return heatmap

    def draw_heatmap(self, image=None, factor=0.5):
        heatmap_min = self.heatmap.min()
        heatmap_max = self.heatmap.max()
        norm_heatmap = 255.0 * (self.heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        color_heatmap = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        if image is not None:
            self.img_height, self.img_width = image.shape[:2]

            # Resize and combine it with the RGB image
            color_heatmap = cv2.resize(color_heatmap, (self.img_width, self.img_height))
            color_heatmap = cv2.addWeighted(image, factor, color_heatmap, (1 - factor), 0)

        return color_heatmap
