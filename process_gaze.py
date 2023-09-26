import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from utils.train_utils import KLDivergence, CC


class ResizeMetrics(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        img = img.resize((40, 30), resample=Image.BILINEAR)
        return {'image': img}


class ToTensorMetrics(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img = sample['image']
        img = self.tensor(img)
        return {'image': img}


image_root = './dataset/val/gaze'
total_gaze = os.listdir(image_root)
gaze_root = './pseudo_labels/ml_predict'

kld_metric = KLDivergence()
cc_metric = CC()
for image in total_gaze:
    transform = transforms.Compose([ResizeMetrics(224), ToTensorMetrics()])
    image_path = os.path.join(image_root, image)
    img = Image.open(image_path)
    img.convert('L')
    sample = {'image': img}
    sample = transform(sample)
    img = sample['image'].unsqueeze(0)

    # gaze_path = os.path.join(gaze_root, image)
    # gaze = Image.open(gaze_path)
    # gaze.convert('L')
    # sample = {'image': gaze}
    # sample = transform(sample)
    # gaze = sample['image'].unsqueeze(0)

    a = torch.zeros_like(img)

    kld_metric.update(a, img)
    cc_metric.update(a, img)
    print(image)

kld_info = kld_metric.compute()
cc_info = cc_metric.compute()

print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
