import torch.nn as nn
import torch
from torchvision.models import alexnet, swin_t
from torchvision import models
import torch.nn.functional as F
from models.convlstm import ConvLSTM
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2


class VisualFP(nn.Sequential):
    def __init__(self):
        super(VisualFP, self).__init__(
            nn.Conv2d(256, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )


class BDDA(nn.Module):
    def __init__(self):
        super(BDDA, self).__init__()
        self.backbone = nn.Sequential(*list(alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).children())[:-2])
        # self.backbone = nn.Sequential(*list(swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1).children())[:-2])
        self.feature_p = VisualFP()
        self.conv_lstm = ConvLSTM(8, hidden_dim=[64, 64, 1],
                                  kernel_size=(3, 3), num_layers=3, batch_first=True,
                                  bias=True, return_all_layers=False)
        self.GB = transforms.GaussianBlur(13, 1.5)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x1 = F.interpolate(x, (36, 64))
        x2 = self.feature_p(x1)
        x3 = self.conv_lstm(x2.unsqueeze(0))[0][0][0]
        y = self.GB(x3)
        y = self.conv(y)
        # y = torch.softmax(y.flatten(start_dim=1) * 255, dim=1).unflatten(1, (1, 36, 64))
        # x = torch.sigmoid(x)
        # for i in range(len(x)):
        #     x[i] = torch.softmax(x[i].flatten(), dim=0).reshape([1, 36, 64])
        # l = x[0].cpu().detach().numpy()
        # l = l.transpose(1, 2, 0) * 255
        # l = l.astype(numpy.uint8)
        # cv2.imshow("l", l)
        # cv2.waitKey()

        self.heatmap = self.process_output(x)

        return y, x1, x2, x3

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


if __name__ == '__main__':
    model = BDDA()
    t = torch.randn(6, 3, 576, 1024)
    y = model(t)

