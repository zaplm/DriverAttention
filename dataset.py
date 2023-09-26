import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torch.utils.data import Dataset
from PIL import Image
import cv2


class BDDADataSet(Dataset):
    def __init__(self, root: str, mode: str, alpha: float = 0.01, test_name: str = ''):
        self.alpha = alpha
        self.mode = mode
        assert os.path.exists(root), f"path '{root}' does not exists."
        self.p_root = './pseudo_labels'
        if mode == 'train':
            self.camera_root = os.path.join(root, "train", "camera")
            self.gaze_root = os.path.join(root, "train", "gaze")
            self.gaze_names = [p for p in os.listdir(self.gaze_root)]

        elif mode == 'train_u':
            self.camera_root = os.path.join(root, "train", "camera")
            self.mask_root = os.path.join(root, 'train', 'masks')
            self.mask_dic = os.listdir(self.mask_root)
            self.mask_path = [os.listdir(os.path.join(self.mask_root, dic)) for dic in self.mask_dic]
            self.p_dic = os.listdir(self.p_root)
            self.p_path = [os.listdir(os.path.join(self.p_root, dic)) for dic in self.p_dic]
            # prior = np.ones((3, 3)) + 1
            # prior[1][1] = 0.5
            # self.prior = cv2.resize(prior, (224, 224), interpolation=cv2.INTER_AREA)
            # self.prior = torch.FloatTensor(self.prior)

        elif mode == 'val':
            self.camera_root = os.path.join(root, "val", "camera")
            self.gaze_root = os.path.join(root, "val", "gaze")
            self.gaze_names = [p for p in os.listdir(self.gaze_root)]

        elif mode == 'test':
            self.camera_root = os.path.join(root, "test", "camera"+test_name)
            self.gaze_root = os.path.join(root, "test", "gaze"+test_name)
            self.gaze_names = [p for p in os.listdir(self.gaze_root)]

        self.video_names = [p for p in os.listdir(self.camera_root)]

        if mode == 'train':
            self.video_path = [os.path.join(self.camera_root, n) for n in self.video_names]
            self.gaze_path = [os.path.join(self.gaze_root, n) for n in self.gaze_names]
        elif mode == 'train_u':
            self.video_path = [os.path.join(self.camera_root, n) for n in self.video_names]
        elif mode == 'val':
            self.video_path = [os.path.join(self.camera_root, n) for n in self.video_names]
            self.gaze_path = [os.path.join(self.gaze_root, n) for n in self.gaze_names]
        elif mode == 'test':
            self.video_path = [os.path.join(self.camera_root, n) for n in self.video_names]
            self.gaze_path = [os.path.join(self.gaze_root, n) for n in self.gaze_names]

    def __getitem__(self, i):
        img = self.convert(self.video_path[i], True)
        if self.mode == 'train_u':
            p = []

            for index, pseudo in enumerate(self.p_path):
                ps = self.convert(os.path.join(self.p_root, self.p_dic[index], pseudo[i]))
                # ps = ps * (mask + word + 0.5)
                # ps /= ps.max()
                if self.alpha != 0.01:
                    ms = []
                    for j, mask in enumerate(self.mask_path):
                        m = self.convert(os.path.join(self.mask_root, self.mask_dic[j], mask[i]))
                        ms.append(m)
                        # ps = torch.cat([ps, m], dim=0)
                    mall = sum(ms)
                    ps = ps * (mall + self.alpha)
                    ps /= ps.max()
                p.append(ps)

            return img, p
        gaze = self.convert(self.gaze_path[i])

        return img, gaze

    def __len__(self):
        return len(self.video_path)

    @staticmethod
    def convert(x_path, is_rgb=False):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        x = Image.open(x_path)
        if is_rgb:
            x = x.convert('RGB')
            x = transform(x)
        else:
            x = x.convert('L')
            x = np.array(x)
            x = x.astype('float')
            x = cv2.resize(x, (224, 224))
            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)

        return x

    def collate_fn(self, batch):
        if self.train:
            images, targets, p = list(zip(*batch))
        else:
            images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        if self.train:
            batched_p = []
            for index in range(len(p[0])):
                temp = [ps[index] for ps in p]
                batched_p.append(cat_list(temp, fill_value=0))

            return batched_imgs, batched_targets, batched_p
        else:
            return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = BDDADataSet("./dataset", train=True)
    print(len(train_dataset))

    val_dataset = BDDADataSet("./dataset", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
