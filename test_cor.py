'''
test the corruption robustness
'''
import os
import argparse
import torch
from dataset.DrDataset import DrDataset
from dataset.SceneDatasetCor import SceneDatasetCor
from torch.utils.data import DataLoader
import utils.train_utils as utils

from utils.train_and_evaluate import evaluate
import csv
# from metann import Learner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='uncertainty-m', help="uncertainty-r/-c, resnet, ConvNext")
    parser.add_argument('--data-path', default='./dataset', help='where to store data')
    parser.add_argument('--save_model', default='5_layer')
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_aucs', default=False, type=bool)

    # parser.add_argument('--severity', default=0, type=int)
    # parser.add_argument('--cam_subdir', default='camera', type=str)


    # parser.add_argument('--output_folder', default='output', type=str)

    return parser.parse_args()


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


def main(args):

    results = {}
    cors = ['gaussian_noise', None, 'snow', 'fog', 'motion_blur', 'impulse_noise', 'jpeg_compression']
    # cors = ['motion_blur']
    # cors = ['gaussian_noise']
    

    for cor in cors:

        dataset = SceneDatasetCor(args.data_path, mode='test', noise_type=cor)
        print(len(dataset))
        data_loader = DataLoader(dataset,
                                batch_size=32,  # must be 1
                                num_workers=8,
                                pin_memory=True)



        if args.model == 'uncertainty-m':
            from models.model import Model
            model = Model('mobileViT', input_dim=args.input_dim)
        else: raise NotImplementedError


        # import pdb; pdb.set_trace()
        checkpoint = torch.load('./save_weights/model_best_{}.pth'.format(args.save_model))
        
        # only use APB
        state_dict = checkpoint['model']
        prefixes_to_remove = ['ub1', 'ub2', 'ub3']
        keys_to_remove = [key for key in state_dict if any(key.startswith(prefix) for prefix in prefixes_to_remove)]
        for key in keys_to_remove:
            del state_dict[key]
        model.load_state_dict(state_dict, strict=False)


        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.to('cuda')

        # kld_metric, cc_metric = evaluate(args, model, data_loader, device='cuda')
        kld_metric, cc_metric = evaluate_batch(args, model, data_loader, device='cuda')
        
        
        kld_info, cc_info = kld_metric.compute(), cc_metric.compute()
        results[cor] = {'kld': kld_info, 'cc': cc_info}
        print(f"val_kld: {kld_info:.3f} val_cc: {cc_info:.3f}")

    print("\nFinal Results:")
    for cor, metrics in results.items():
        if cor is None:
            print(f"clean: {metrics}")
        else:
            print(f"{cor}: {metrics}")

    with open(f'{args.save_model}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(['corruption', 'kld', 'cc'])

        for cor, metrics in results.items():
            cor_name = 'clean' if cor is None else cor
            writer.writerow([cor_name, metrics['kld'], metrics['cc']])

    print("Results have been written to results.csv")


if __name__ == '__main__':
    args = parse_args()
    main(args)
