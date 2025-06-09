import os
import argparse
# from dataset.BDDADataset import BDDADataSet
# from dataset.RadiateDataset import RadiateDataset
from torch.utils import data
from dataset.BddoiaDataset import BddoiaDataset
import numpy as np


from utils.train_and_evaluate import train_dec_one_epoch, evaluate, get_params_groups, create_lr_scheduler
# import matplotlib.pyplot as plt
# import wandb
import torch
import datetime
import time
import wandb
from tqdm import tqdm

# from model import (generate_model, load_pretrained_model, make_data_parallel,
#                    get_fine_tuning_parameters)
# from models.strg import STRG
# from models.rpn import RPN
torch.manual_seed(3407)
# wandb.init(project='fixation')




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
    # parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c/-m, resnet, ConvNext")
    parser.add_argument('--atten_model', default='init', help="base atten model")

    parser.add_argument('--input_channel', default=1, type=int)
    # parser.add_argument('--alpha', default=-1, type=float, help="if alpha=-1, without mask")
    # parser.add_argument('--loss_func', default='kld', help='bce/ce')
    # parser.add_argument('--val_aucs', default=False, type=bool)
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--use_wandb", action='store_true',
            help="Use wandb to record")
    parser.add_argument('--project_name', default='prj', help="wandb project name")
    parser.add_argument('--name', default='', help="save_name")
    parser.add_argument('--pretrain', default='dr', help="pretrain from attention")
    parser.add_argument('--no_weight', action='store_true', help="weather use atten_weight")
    parser.add_argument("--nrois", default=12, type=int)



    args = parser.parse_args()

    return args


def main(args):


    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            name = args.name,
            config={
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size
            }
        )

    #args.model
    from models.DecisionModel import Model
    model = Model('mobileViT', nrois=args.nrois, no_weight=args.no_weight).cuda()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    train_dataset = BddoiaDataset(args.data_path, mode='train', gaze_dir=args.atten_model)


    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True)
    print(len(train_dataset))
    print(len(train_data_loader))



    

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)


    model = model.to(device)
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        loss, lr = train_dec_one_epoch(args, model, optimizer, train_data_loader, device, epoch, lr_scheduler,
                                        print_freq=args.print_freq, scaler=None)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        torch.save(save_file, "save_weights/model_best_{}_{}.pth".format(args.name, epoch))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))

if __name__ == '__main__':
    a = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(a)
