'''
test the corruption robustness
'''

import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import random
from tqdm import tqdm


import os
import argparse
import torch
from dataset.StatHard import StatHard
from dataset.SceneDataset import SceneDataset
from torch.utils.data import DataLoader
import utils.train_utils as utils

from utils.train_and_evaluate import evaluate
import csv
import cv2
# from metann import Learner


def evaluate_batch(args, model, data_loader, device):
    # import pdb; pdb.set_trace()
    model.eval()
    kld_metric = utils.KLDivergence()
    cc_metric = utils.CC()
    if args.val_aucs:
        aucs_metric = utils.SAuc()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        os.makedirs('./output', exist_ok=True)
        count = 0
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            if args.model.find('uncertainty') != -1:
                output = model(images)
            else:
                output, _ = model(images)
            batch_size = images.size(0)
            for i in range(batch_size):
                kld_metric.update(output[i].unsqueeze(0), targets[i].unsqueeze(0))
                cc_metric.update(output[i].unsqueeze(0), targets[i].unsqueeze(0))
                if args.val_aucs:
                    aucs_metric.update(output[i].unsqueeze(0), targets[i].unsqueeze(0))
    if args.val_aucs:
        return kld_metric, cc_metric, aucs_metric
    else:
        return kld_metric, cc_metric

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data-path', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--sample_list', default='hard_cases_gl4.txt')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)

   

    return parser.parse_args()

def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    # c = s_map.size(1)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 1e-8
    result = gt * torch.log(eps + gt / (s_map + eps))
    return torch.mean(torch.sum(result, 1))


def convert(x_path, is_rgb=False, resize=True):
    transform_with_resize = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the shorter side to 224, maintaining aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    transform_wo_resize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    x = Image.open(x_path)
    w, h = x.size
    # print(x.size)

    if is_rgb:
        x = x.convert('RGB')
        if ( w == 224 and h == 224):
            x = transform_wo_resize(x)
        else:
            x = transform_with_resize(x)
    else:
        # print('not rgb')
        x = x.convert('L')
        x = np.array(x)
        x = x.astype('float')
        
        if ( not (w == 224 and h == 224 ) ) and resize==True:
            x = cv2.resize(x, (224, 224))

        if np.max(x) > 1.0:
            x = x / 255.0
        x = torch.FloatTensor(x).unsqueeze(0)

    return x


def test(args):
    from models.model import Model
    model = Model('mobileViT', input_dim=args.input_dim)

    checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
    model.load_state_dict(checkpoint['model'])
    model = model.to('cuda')
    
    # model.eval()
    # kld_metric = utils.KLDivergence()
    # cc_metric = utils.CC()

def main(args):

  
    hard_cases_prefix = 'hard_cases_gl'
    # import pdb; pdb.set_trace()
    for thresh_hold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        sample_list = hard_cases_prefix + str(6 - thresh_hold) + '.txt'
        dataset = StatHard(args.data_path, sample_list = sample_list)
        data_loader = DataLoader(dataset,
                                batch_size=32, 
                                num_workers=8,
                                pin_memory=True)
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_dim)
        checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
        model.load_state_dict(checkpoint['model'])
        model = model.to('cuda')
        kld_metric, cc_metric = evaluate_batch(args, model, data_loader, device='cuda')

        
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        print('finish eval for the threshold: {}'.format(thresh_hold))
        print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")



if __name__ == '__main__':
    args = parse_args()
    main(args)

