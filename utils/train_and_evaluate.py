import os

import torch
import torch.nn as nn
import math
import utils.train_utils as utils
from torch.nn import functional as F
from torch import autograd
# import wandb
import numpy as np
import cv2
import wandb
from pathlib import Path

def mixup_data(x, pseudos,  alpha=1., lam=None,  use_cuda=True):
    '''random select a data in batch and mix it
    alpha:beta 分布的参数需要调整，影响比较大
    '''
    if not lam:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mix_data = lam*x + (1-lam)*x[index, :]
    mix_data = torch.cat([x, mix_data], dim=0)

    for i, pseudo in enumerate(pseudos):
        new_pseudo = lam * pseudo + (1-lam) * pseudo[index, :]
        pseudos[i] = torch.cat([pseudo, new_pseudo], dim=0)

    return mix_data, pseudos, lam

def criterion(inputs, p, e, type='bce'):
    total = []
    # kld = nn.KLDivLoss(reduction='none')
    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')
    for i in range(len(p)):
        if type == 'bce':
            # p[i] = torch.split(p[i], dim=1, split_size_or_sections=1)[0]
            bce_loss = bce(inputs, p[i])
            loss = bce_loss * torch.exp(-e[i]) + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        elif type == 'mse':
            mse_loss = mse(inputs, p[i])
            loss = mse_loss * torch.exp(-e[i]) / 2 + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        elif type == 'ce':
            ce_loss = ce(inputs, p[i])
            loss = ce_loss * torch.exp(-e[i]) + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        elif type == 'kld':
            kld_loss = kldiv(inputs.squeeze(1), p[i].squeeze(1))
            loss = kld_loss * torch.exp(-e[i]) + e[i] / 2
            loss = torch.sum(loss, dim=[1, 2, 3])
        # loss = [l(inputs[i], p[j][i][0].unsqueeze(0)) * torch.exp(-e[j][i]) + e[j][i] / 2 for i in range(len(inputs))]
        # branch = [l(s[j][i], p[j][i]) * torch.exp(-e[j][i]) + e[j][i] / 2 for i in range(len(inputs))]
        total.append(loss)
    total = sum(total)
    total = total.mean()
    return total


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
    # return result.reshape(batch_size, w, h)


def full(pred, gt):
    loss = kldiv(pred, gt)
    return loss


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


def evaluate(args, model, data_loader, device):
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
            kld_metric.update(output, targets)
            cc_metric.update(output, targets)
            if args.val_aucs:
                aucs_metric.update(output, targets)
    if args.val_aucs:
        return kld_metric, cc_metric, aucs_metric
    else:
        return kld_metric, cc_metric
    
    
def train_robo_cor_one_epoch(args, model, optimizer, data_loader, val_data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    if not hasattr(train_robo_cor_one_epoch, "iter_counter"):
        train_robo_cor_one_epoch.iter_counter = 0  
    if not hasattr(train_robo_cor_one_epoch, "current_cc"):
        train_robo_cor_one_epoch.current_cc = 0.  

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if args.model.find('uncertainty') != -1:
        # count = 0
        for image, p in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device)
            p = [ps.to(device) for ps in p]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output, e = model(image, p)
                loss = criterion(output, p, e, args.loss_func)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()

            # optimizer.step()
            optimizer.zero_grad()

            # lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
            train_robo_cor_one_epoch.iter_counter += 1  
            if train_robo_cor_one_epoch.iter_counter % 1000 == 0:
                save_file = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args}

                
                kld_metric, cc_metric = evaluate_batch(args, model, val_data_loader, device=device)
                kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
                print(f"[epoch: {epoch}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
                
                if val_snow_data_loader is not None:
                    kld_snow_metric, cc_snow_metric = evaluate_batch(args, model, val_snow_data_loader, device=device)
                    kld_snow_info, cc_snow_info = kld_snow_metric.compute(), cc_snow_metric.compute()
                    print(f"[epoch: {epoch}] val_kld: {kld_snow_info:.3f} val_cc: {cc_snow_info:.3f}")
                    
                if val_gau_data_loader is not None:
                    kld_gau_metric, cc_gau_metric = evaluate_batch(args, model, val_gau_data_loader, device=device)
                    kld_gau_info, cc_gau_info = kld_gau_metric.compute(), cc_gau_metric.compute()
                    print(f"[epoch: {epoch}] val_kld: {kld_gau_info:.3f} val_cc: {cc_gau_info:.3f}")

                torch.save(save_file, Path(args.save_dir)/"model_best_{}_{}_{}_{}_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info), "{:.5f}".format(cc_snow_info), "{:.5f}".format(kld_snow_info), "{:.5f}".format(cc_gau_info), "{:.5f}".format(kld_gau_info))) 
            

    return metric_logger.meters["loss"].global_avg, metric_logger.meters["lr"].global_avg


def train_one_epoch(args, model, optimizer, data_loader, val_data_loader, val_snow_data_loader,  device, epoch, lr_scheduler, print_freq=10, scaler=None):
    if not hasattr(train_one_epoch, "current_cc"):
        train_one_epoch.current_cc = 0.  

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # import pdb; pdb.set_trace()
    if args.model.find('uncertainty') != -1:
        # import pdb; pdb.set_trace()
        for image, p in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device)
            p = [ps.to(device) for ps in p]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output, e = model(image, p)
                loss = criterion(output, p, e, args.loss_func)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()

            # optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
            
        save_file = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args}

        kld_metric, cc_metric = evaluate_batch(args, model, val_data_loader, device=device)
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        print(f"[epoch: {epoch}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
        
        if(args.use_wandb):
            wandb.log({'lr': lr, 
                    'loss': loss, 

                    'cc': cc_info, 
                    'kld': kld_info,
                    'len dataset': len(data_loader)*args.batch_size
                    })

        if train_one_epoch.current_cc  <=cc_info:
            torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
            train_one_epoch.current_cc  = cc_info
            

    return metric_logger.meters["loss"].global_avg, metric_logger.meters["lr"].global_avg


def train_lt_one_epoch(args, model, optimizer, data_loader, val_data_loader, val_snow_data_loader,  device, epoch, lr_scheduler, print_freq=10, scaler=None):
    if not hasattr(train_lt_one_epoch, "iter_counter"):
        train_lt_one_epoch.iter_counter = 0 
    if not hasattr(train_lt_one_epoch, "current_cc"):
        train_lt_one_epoch.current_cc = 0. 

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if args.model.find('uncertainty') != -1:
        for image, image_c, p, p_c in metric_logger.log_every(data_loader, print_freq, header):
            
            image = image.to(device)
            p = [ps.to(device) for ps in p]
        
            image_c = image_c.to(device)
            p_c = [ps.to(device) for ps in p_c]

            image, p, lam = mixup_data(image, p, alpha=10)
            image_c, p_c, _ = mixup_data(image_c, p_c, lam=lam, alpha=10.)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output, e = model(image, p)
                output_c, e_c = model(image_c, p_c)
                
                loss = criterion(output, p, e, args.loss_func)
                loss_c = criterion(output_c, p_c, e_c, args.loss_func)
                if not torch.isnan(loss_c):
                    loss  = loss + loss_c
                    

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()

            # optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
            
        save_file = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args}

        kld_metric, cc_metric = evaluate_batch(args, model, val_data_loader, device=device)
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        print(f"[epoch: {epoch}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
        
        if train_lt_one_epoch.current_cc  <=cc_info:
            torch.save(save_file, "save_weights/model_best_{}_{}_{}.pth".format(args.name, "{:.5f}".format(cc_info), "{:.5f}".format(kld_info) ) )
            train_lt_one_epoch.current_cc  = cc_info
            

    return metric_logger.meters["loss"].global_avg, metric_logger.meters["lr"].global_avg

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子
        注意在训练开始之前 pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def create_stage_lr_scheduler(optimizer,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据epoch返回一个学习率倍率因子
        """

        if warmup is True and x <= warmup_epochs:
            alpha = x / warmup_epochs
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = x - warmup_epochs
            cosine_steps = epochs - warmup_epochs
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
