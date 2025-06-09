import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image
from scipy.ndimage import zoom
import json
import random
import cv2
from pathlib import Path

# from maskrcnn_benchmark.data.transforms import transforms as T


class BddoiaDataset(Dataset):
    def __init__(self, root, mode, gaze_dir):
        super(BddoiaDataset, self).__init__()
        self.root = Path(root)
        self.mode = mode
        self.gaze_dir = gaze_dir
        os.makedirs(self.root/self.gaze_dir, exist_ok=True)
        self.imageRoot = str( self.root / 'data')
        if self.mode == 'prepare':
            self.imgNames = [os.path.join(self.imageRoot, file) for file in os.listdir(self.imageRoot)]
            self.count = len(self.imgNames)
            print("number of samples in dataset:{}".format(self.count))
            self.perm = list(range(self.count))
            random.shuffle(self.perm)
            return
        if self.mode == 'infer_example':
            prefix = 'test'
        else:
            prefix = self.mode 
        self.gtRoot = str( self.root/ (prefix + '_25k_images_actions.json'))
        self.reasonRoot = str( self.root / (prefix + '_25k_images_reasons.json'))


        with open(self.gtRoot) as json_file:
            data = json.load(json_file)
        with open(self.reasonRoot) as json_file:
            reason = json.load(json_file)
        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            # print(len(action_annotations[ind]['category']))
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = osp.join(self.imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))
        self.perm = list(range(self.count))
        random.shuffle(self.perm)

    def __len__(self):
        return self.count

    @staticmethod
    def convert(x_path, is_rgb=False):
        # print(x_path)
        transform_with_resize = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        x = Image.open(x_path)
        w, h = x.size

        if is_rgb:
            x = x.convert('RGB')
            if ( w == 224 and h == 224):
                x = transform_wo_resize(x)
            else:
                x = transform_with_resize(x)
        else:
            x = x.convert('L')
            x = np.array(x)
            x = x.astype('float')
            
            if ( w != 224 or h != 224):
                x = cv2.resize(x, (224, 224))

            if np.max(x) > 1.0:
                x = x / 255.0
            x = torch.FloatTensor(x).unsqueeze(0)
            # print(x.shape)

        return x
    def img2gazeName(self, var):
        parts = var.split('/')
        parts[-2] = self.gaze_dir
        res = '/'.join(parts)   
        return res  

    def __getitem__(self, ind):
        # test = True
        imgName = self.imgNames[self.perm[ind]]
        # if self.mode == 'infer_example' and imgName.split('/')[-1] != '39337da8-b0a8bc14_3.jpg':
        #     return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), 'dump'
            
        img = self.convert(imgName, True)
        gazeName = self.img2gazeName(imgName)

        if self.mode == 'prepare':
            return img, gazeName
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        reason = np.array(self.reasons[self.perm[ind]], dtype=np.int64)
        gaze = self.convert(gazeName)
        if self.mode == 'infer_example':
            return img, gaze, torch.FloatTensor(target)[0:4], torch.FloatTensor(reason), imgName
        else:
            return img, gaze, torch.FloatTensor(target)[0:4], torch.FloatTensor(reason)


if __name__ == '__main__':
    dataset = BddoiaDataset('/data_2/bxy/attn_autodl_backup/bddoia', mode='prepare', gaze_dir='bd_beta')
    import pdb; pdb.set_trace()
    img, gaze_path = dataset[0]
    # img, action, reason = dataset[0]

