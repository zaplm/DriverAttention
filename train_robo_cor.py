import os
import argparse
from dataset.SceneDataset import SceneDataset
from dataset.MixDataset import MixDataset

from torch.utils import data
from torch.utils.data import  ConcatDataset

from utils.train_and_evaluate import train_robo_cor_one_epoch, evaluate, get_params_groups, create_stage_lr_scheduler
import wandb
import torch
import datetime
import time
from tqdm import tqdm
# from metann import Learner
import utils.train_utils as utils
import cv2
import csv
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

torch.manual_seed(3407)
    
def run_infer(args, model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        count = 0
        for images, out_path in metric_logger.log_every(data_loader, 100, header):
            images =  images.to(device)
            # images = images.unsqueeze(0)
            if args.model.find('uncertainty') != -1:
                outputs = model(images)
            else:
                raise(NotImplementedError)
            for i in range(images.shape[0]):
                output = outputs[i]
                out = (output / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                out = out.astype(np.uint8)
                cv2.imwrite(str(out_path[i]), out)
                count += 1

def sort_desc_with_index(lst):
    sorted_list_with_index = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    sorted_indexes, sorted_values = zip(*sorted_list_with_index)
    return list(sorted_values), list(sorted_indexes)

def tensor2img(x:torch.Tensor):
    '''return a img so that cv2 can save it'''
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    x = x.cpu().detach().permute(1, 2, 0).numpy()
    x = (x * 255).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    return x

def mixup_data(x, p, attention, idx1, idx2):
    '''random select a data in batch and mix it
    attenttion is better not divide the max
    '''
    x1, x2 = x[idx1], x[idx2]
    # p1, p2 = p[idx1], p[idx2]
    atten1, atten2 = attention[idx1], attention[idx2]


    mix_atten_sum =  atten1 + atten2 
    eps = 1e-7
    mix_data = (x1 * (atten1 + eps) + x2 * (atten2 + eps)) / (mix_atten_sum + 2*eps)
    mix_p = []
    for pi in p:
        p1, p2 = pi[idx1], pi[idx2]
        pseudo = (p1 * (atten1 + eps) + p2 * (atten2 + eps)) / (mix_atten_sum + 2*eps)
        mix_p.append(pseudo)
    return mix_data, mix_p


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
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))

def run_infer_mixup(args, model, data_loader, gaze_average, device, topK=8):
    model.eval()
    gaze_average = gaze_average.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    os.makedirs(str(Path(args.data_path)/args.mix_dir), exist_ok=True)
    os.makedirs(str(Path(args.data_path)/args.mix_dir/'0'), exist_ok=True)
    os.makedirs(str(Path(args.data_path)/args.mix_dir/'1'), exist_ok=True)
    os.makedirs(str(Path(args.data_path)/args.mix_dir/'camera_224_224'), exist_ok=True)

    

    with torch.no_grad():
        count = 0
        for images, p, out_path in metric_logger.log_every(data_loader, 100, header):
            batch_size = images.size()[0]
            images =  images.to(device)
            p = [ps.to(device) for ps in p]
            if args.model.find('uncertainty') != -1:
                outputs = model(images)
            else:
                raise(NotImplementedError)
            
            kl_list = []
            attention_list = []
            for i in range(images.shape[0]):
                #trival infer
                output = outputs[i]
                attention_list.append(output)
                out = (output / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                # cv2.imwrite(str(out_path[i]), out)

                #cal kl 
                kl = kldiv(output, gaze_average).item()
                kl_list.append(kl)
            if (batch_size != args.batch_size):
                continue
            _,  idxs = sort_desc_with_index(kl_list)
            top_kl_idx = idxs[:topK]
            for idx in top_kl_idx:
                for j in range(batch_size):
                    mix_imgs, mix_ps =  mixup_data(images, p, attention_list, idx, idxs[j])
                    img_out = tensor2img(mix_imgs)
                    for p_idx, mix_p in enumerate(mix_ps):
                        p_out = (mix_p / mix_p.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                        p_out = p_out.astype(np.uint8)
                        cv2.imwrite( str(Path(args.data_path)/args.mix_dir/'{}'.format(p_idx)/"{}.jpg".format(count)), p_out) 

                    cv2.imwrite( str(Path(args.data_path)/args.mix_dir/'camera_224_224'/"{}.jpg".format(count)), img_out)
                    count += 1


def parse_args():
    parser = argparse.ArgumentParser(description="new model training")
    parser.add_argument("--data-path", default="./dataset", help="BDDA root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=1, type=int, help="validation interval default 10 Epochs")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # parser.add_argument('--resume', default='./save_weights/model_best_kldd3.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c/-m, resnet, ConvNext")
    parser.add_argument('--input_channel', default=1, type=int)
    parser.add_argument('--alpha', default=0.3, type=float, help="if alpha=-1, without mask")
    parser.add_argument('--project_name', default='prj', help="wandb project name")
    parser.add_argument('--name', default='', help="save_name")
    parser.add_argument('--loss_func', default='kld', help='bce/ce')
    parser.add_argument('--val_aucs', default=False, type=bool)
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--use_wandb", action='store_true',
                    help="Use wandb to record")
    parser.add_argument('--p_dic', default=['ml_p', 'unisal_p'], nargs='+', help='A list of pseudoss')
    
    
    #==================for aug strategy===============================
    parser.add_argument('--mix_dir', default='mix_data', help="dir to save the mixupdata")
    parser.add_argument('--topK', default=8, type=int)
    
    
    
    # =======additional ===========
    parser.add_argument("--save-dir", default="save_weights", help="BDDA root")
    parser.add_argument('--shuffle', default=0, type=int)
    
    
    
    
    args = parser.parse_args()

    return args


def write_csv(name, epoch, kl_db):
    csv_file_path = './csv/' + name  + '_{}'.format(epoch) + '.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])  
        writer.writerows(list(kl_db.items())) 

def main(args):
        

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    val_dataset = SceneDataset(args.data_path, mode='val')
    
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=args.batch_size, 
                                      num_workers=num_workers,
                                      pin_memory=True)
    
    
    print('infer mixup data in: {}'.format(args.mix_dir))
    print('top K: {}'.format(args.topK))
    print('data loader workers number: %d' % num_workers)
    print('length of val dataset: %d' % len(val_dataset))
    print('length of train_batch_size: %d' % batch_size )
    print(args.data_path)

    
    if args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_channel)
    else: raise NotImplementedError
    model = model.to(device)


    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_stage_lr_scheduler(optimizer, args.epochs,
                                    warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    start_time = time.time()
    mix_train_dataset = None
    init_train_dataset = SceneDataset(args.data_path, mode='train', p_dic = args.p_dic, infer_gaze_subdir=args.mix_dir)
    

    start_time = time.time()
    model = model.to(device)

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        if epoch == 0:
            train_dataset = init_train_dataset
            train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True)
        else:
            assert(mix_train_dataset is not None)
            train_dataset = ConcatDataset([init_train_dataset, mix_train_dataset])
            train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True)
    
        print('length of  train_dataset: %d' % len(train_dataset))
        loss, lr = train_robo_cor_one_epoch(args, model, optimizer, train_data_loader, val_data_loader, device, epoch, lr_scheduler,
                                        print_freq=args.print_freq, scaler=None)
        lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()



        if mix_train_dataset is not None:
            mix_train_dataset.clear()
        '''
        init mixup and infer the distribution
        '''
        print('infering the dataset to get distirbution')
        init_infer_dataset= SceneDataset(args.data_path, mode='infer',
                                         out_folder=args.mix_dir, infer_gaze_subdir=args.mix_dir)
        infer_dataloader = data.DataLoader(init_infer_dataset,
                            batch_size=args.batch_size,  
                            num_workers=num_workers,
                            pin_memory=True)    
        run_infer(args, model, infer_dataloader, device='cuda') 
        kl_db, gaze_average = init_train_dataset.cal_data_kl(name=args.name)
        write_csv(args.name, epoch, kl_db)

        
        '''
        mix dataset infer and get data for next time
        '''
        print('generating mixup data')
        init_infer_dataset = SceneDataset(args.data_path, mode='infer_mix', p_dic = args.p_dic)
        init_infer_dataloader = data.DataLoader(init_infer_dataset,
                                    batch_size=args.batch_size,  
                                    num_workers=num_workers,
                                    shuffle=args.shuffle,
                                    pin_memory=True)
        run_infer_mixup(args, model, init_infer_dataloader, gaze_average, device='cuda', topK = args.topK) 
        print('infering the mixup data')
        mix_infer_dataset= MixDataset(args.data_path, mode='infer', mix_dir=args.mix_dir)
        mix_infer_dataloader = data.DataLoader(mix_infer_dataset,
                            batch_size=args.batch_size,  
                            num_workers=num_workers,
                            pin_memory=True)    
        run_infer(args, model, mix_infer_dataloader, device='cuda') 

        print('select the mixup data')
        mix_train_dataset= MixDataset(args.data_path, mode='train', mix_dir=args.mix_dir, p_dic=['0'] if len(args.p_dic) == 1 else ['0', '1'])
        kl_db = mix_train_dataset.get_data_by_kl(kl_db, gaze_average)
        write_csv(args.name + 'aug', epoch, kl_db)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))


if __name__ == '__main__':
    a = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(a)
