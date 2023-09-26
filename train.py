import argparse
from dataset import BDDADataSet
from torch.utils import data
from utils.train_and_evaluate import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
import matplotlib.pyplot as plt
# import wandb
import torch
import datetime
import time
import os
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
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c/-m, resnet, ConvNext")
    parser.add_argument('--input_channel', default=1, type=int)
    parser.add_argument('--alpha', default=0.01, type=float, help="if alpha=0.01, without mask")
    parser.add_argument('--name', default='', help="save_name")
    parser.add_argument('--loss_func', default='kld', help='bce/ce')
    parser.add_argument('--val_aucs', default=False, type=bool)
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if args.model.find('uncertainty') != -1:
        train_dataset = BDDADataSet(args.data_path, mode='train_u', alpha=args.alpha)
    else:
        train_dataset = BDDADataSet(args.data_path, mode='train')
    val_dataset = BDDADataSet(args.data_path, mode='val')

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=False,
                                        pin_memory=True)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True)

    if args.model == 'uncertainty-c':
        from models.model import Model
        model = Model('ConvNext', input_dim=args.input_channel).cuda()

    elif args.model == 'uncertainty-r':
        from models.model import Model
        model = Model('resnet', input_dim=args.input_channel).cuda()

    elif args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_channel).cuda()

    elif args.model == 'uncertainty-mn':
        from models.model import Model
        model = Model('mobilenet', input_dim=args.input_channel).cuda()

    elif args.model == 'uncertainty-v':
        from models.model import Model
        model = Model('vgg', input_dim=args.input_channel).cuda()

    elif args.model == 'uncertainty-d':
        from models.model import Model
        model = Model('densenet', input_dim=args.input_channel).cuda()

    elif args.model == 'resnet':
        from models.models import ResNetModel
        model = ResNetModel(train_enc=True).cuda()

    elif args.model == 'ConvNext':
        from models.models import ConvNextModel
        model = ConvNextModel().cuda()

    elif args.model == 'mobileViT':
        from models.MobileViT import mobile_vit_small
        model = mobile_vit_small().cuda()
        weights_dict = torch.load('models/mobilevit_s.pt', map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_kld, current_cc = 10.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(args, model, optimizer, train_data_loader, device, epoch, lr_scheduler,
                                        print_freq=args.print_freq, scaler=None)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            kld_metric, cc_metric = evaluate(args, model, val_data_loader, device=device)
            kld_info, cc_info = kld_metric.compute(), cc_metric.compute()

            # wandb.log({'kld': kld_info})
            # wandb.log({'cc': cc_info})
            print(f"[epoch: {epoch}] val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
            # write into txt
            # with open(results_file, "a") as f:
            #     # 记录每个epoch对应的train_loss、lr以及验证集各指标
            #     write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
            #                  f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} kld: {kld_info:.3f} cc: {cc_info:.3f}\n"
            #     f.write(write_info)

            # 当前最佳
            if current_cc <= cc_info:
                torch.save(save_file, "save_weights/model_best_{}.pth".format(args.name))
                current_cc = cc_info

        # 只存在最后十个epoch的model
        # if os.path.exists(f"save_weights/model_{epoch - 10}.pth"):
        #     os.remove(f"save_weights/model_{epoch - 10}.pth")
        #
        # torch.save(save_file, f"save_weights/model_{epoch}.pth")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))


if __name__ == '__main__':
    a = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(a)
