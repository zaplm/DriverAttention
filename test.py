import argparse
import torch
from dataset import BDDADataSet
from torch.utils.data import DataLoader
from utils.train_and_evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobileViT', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data_root', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--test_data', default='bdda', help='dr, da, bdda')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)

    return parser.parse_args()


def main(args):
    if args.test_data == 'bdda':
        dataset = BDDADataSet(args.data_root, mode='test')
    elif args.test_data == 'dr':
        dataset = BDDADataSet(args.data_root, mode='test', test_name='_dr')
    elif args.test_data == 'da':
        dataset = BDDADataSet(args.data_root, mode='test', test_name='_da')

    data_loader = DataLoader(dataset,
                             batch_size=1,  # must be 1
                             num_workers=8,
                             pin_memory=True)

    if args.model == 'uncertainty-c':
        from models.model import Model
        model = Model('ConvNext', input_dim=args.input_dim).cuda()

    elif args.model == 'uncertainty-r':
        from models.model import Model
        model = Model('resnet', input_dim=args.input_dim).cuda()

    elif args.model == 'uncertainty-m':
        from models.model import Model
        model = Model('mobileViT', input_dim=args.input_dim).cuda()

    elif args.model == 'uncertainty-v':
        from models.model import Model
        model = Model('vgg', input_dim=args.input_dim).cuda()

    elif args.model == 'uncertainty-mn':
        from models.model import Model
        model = Model('mobilenet', input_dim=args.input_dim).cuda()
    
    elif args.model == 'uncertainty-d':
        from models.model import Model
        model = Model('densenet', input_dim=args.input_dim).cuda()

    elif args.model == 'resnet':
        from models.models import ResNetModel
        model = ResNetModel(train_enc=True).cuda()

    elif args.model == 'ConvNext':
        from models.models import ConvNextModel
        model = ConvNextModel().cuda()

    elif args.model == 'mobileViT':
        from models.MobileViT import mobile_vit_small
        model = mobile_vit_small().cuda()

    checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
    model.load_state_dict(checkpoint['model'])

    if args.val_aucs:
        kld_metric, cc_metric, aucs_metric = evaluate(args, model, data_loader, device='cuda')
        kld_info, cc_info, aucs_info = kld_metric.compute(), cc_metric.compute(), aucs_metric.compute()

        print(f"val_aucs: {aucs_info:.3f} val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")
    else:
        kld_metric, cc_metric = evaluate(args, model, data_loader, device='cuda')
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()

        print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
