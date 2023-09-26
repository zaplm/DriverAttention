import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath


class CDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_upsampling1 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.linear_upsampling2 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.linear_upsampling3 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 1),
            nn.LayerNorm([24, 320, 192], eps=1e-6)
        )
        self.conv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=7, padding=3, groups=24),
            nn.LayerNorm([24, 320, 192], eps=1e-6),
            nn.Conv2d(24, 24 * 4, 1),
            nn.GELU(),
            nn.Conv2d(24 * 4, 24, 1),
        )
        self.downsample1 = nn.Sequential(
            nn.LayerNorm([24, 320, 192], eps=1e-6),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=2, stride=2),
        )
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding=3, groups=48),
            nn.LayerNorm([48, 160, 96], eps=1e-6),
            nn.Conv2d(48, 48 * 4, 1),
            nn.GELU(),
            nn.Conv2d(48 * 4, 48, 1),
        )
        self.downsample2 = nn.Sequential(
            nn.LayerNorm([48, 160, 96], eps=1e-6),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=2, stride=2),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=7, padding=3, groups=96),
            nn.LayerNorm([96, 80, 48], eps=1e-6),
            nn.Conv2d(96, 96 * 4, 1),
            nn.GELU(),
            nn.Conv2d(96 * 4, 96, 1),
        )
        self.downsample3 = nn.Sequential(
            nn.LayerNorm([96, 80, 48], eps=1e-6),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=2, stride=2),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=7, padding=3, groups=192),
            nn.LayerNorm([192, 40, 24], eps=1e-6),
            nn.Conv2d(192, 192 * 4, 1),
            nn.GELU(),
            nn.Conv2d(192 * 4, 192, 1),
        )
        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=24, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)

    def forward(self, img):
        o1 = self.stem(img)
        o1 = self.conv_layer0(o1) + o1
        o2 = self.downsample1(o1)
        o2 = self.conv_layer1(o2) + o2
        o3 = self.downsample2(o2)
        o3 = self.conv_layer2(o3) + o3
        o4 = self.downsample3(o3)
        o4 = self.conv_layer3(o4) + o4
        x = self.linear_upsampling1(o4)
        x = torch.cat([x, o3], dim=1)
        x = self.deconv_layer0(x)
        x = self.linear_upsampling2(x)
        x = torch.cat([x, o2], dim=1)
        x = self.deconv_layer1(x)
        x = self.linear_upsampling3(x)
        x = torch.cat([x, o1], dim=1)
        x = self.deconv_layer2(x)

        return x


if __name__ == '__main__':
    y = torch.randn((1, 3, 320, 192))
    m = CDNN()
    output = m(y)
