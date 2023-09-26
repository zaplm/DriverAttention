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


def evaluate(args, model, data_loader, device):
    model.eval()
    # mae_metric = utils.MeanAbsoluteError()
    # f1_metric = utils.F1Score()
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
            out = (output[0] / output.max()).permute(1, 2, 0).cpu().detach().numpy() * 255
            out.astype(int)
            cv2.imwrite('./output/{}.jpg'.format(count), out)
            count += 1

            # mae_metric.update(output, targets)
            # f1_metric.update(output, targets)
            kld_metric.update(output, targets)
            cc_metric.update(output, targets)
            if args.val_aucs:
                aucs_metric.update(output, targets)
    if args.val_aucs:
        return kld_metric, cc_metric, aucs_metric
    else:
        return kld_metric, cc_metric


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
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
                # if count == 26:
                #     out = (e[0][0] / e[0][0].max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                #     cv2.imwrite('./{}.jpg'.format(0), out.astype(np.uint8))
                #     out = (e[1][0] / e[1][0].max()).permute(1, 2, 0).cpu().detach().numpy() * 255
                #     cv2.imwrite('./{}.jpg'.format(1), out.astype(np.uint8))
                # count += 1
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
    else:
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output, _ = model(image)
                loss = full(output.squeeze(1), target.squeeze(1))

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10)
                optimizer.step()

            # optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


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
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
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
