import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from dataset.load_val_vfe_inferencebycase import load_valset
from engine_twohead4inference import evaluate
from models import VFE
from evaluation import get_pr_re_acc_jacc, get_map

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# number of trainable params: 43134438
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1000, type=int) # 500 one gpu
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # Model parameters
    parser.add_argument('--data_path', type=str, default='/media/cygzz/data/rtao/data/cholec/cholec80/npys')
    parser.add_argument('--output_dir', default='pred', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0, 1], help='device to use for training / testing')
    parser.add_argument('--seed', default=130, type=int)
    parser.add_argument('--resume', default='weights/stage1-light.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    return parser


def main(args):
    print(args.resume)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = VFE(num_phase_class=7, num_tool_class=7)
    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model.to(device)

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    print("Start inference")
    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    pred_dir =  Path(args.output_dir+'/stage1')
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    if os.path.isdir(pred_dir) == False:
        os.mkdir(pred_dir)

    test_id = []
    for i in range(80):
        if i < 40:
            pass
        else:
            test_id.append('video' + "{0:02d}".format(i + 1))

    accuracy, precision, recall, jaccard, mAPs = [], [], [], [], []
    for file_id in test_id:
        print(file_id)
        dataset_val = load_valset(file_paths=args.data_path, file_id=file_id)
        data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)
        pred_phase_label, gt_phase_label, pred_tool_label, gt_tool_label = evaluate(model, data_loader_val, device)


        acc, pr, re, jacc = get_pr_re_acc_jacc(input=pred_phase_label, label=gt_phase_label)
        map = get_map(input=pred_tool_label, label=gt_tool_label)

        accuracy.append(acc)
        precision.append(pr)
        recall.append(re)
        jaccard.append(jacc)
        mAPs.append(map)
        print('acc:',acc,'pr:',pr,'re:',re,'jacc:',jacc, 'mAP:', map)

    print('-average accuracy:', np.mean(np.array(accuracy)), '-std:', np.std(np.array(accuracy)))
    print('-average precision:', np.mean(np.array(precision)), '-std:', np.std(np.array(precision)))
    print('-average recall:', np.mean(np.array(recall)), '-std:', np.std(np.array(recall)))
    print('-average JACC score:', np.mean(np.array(jaccard)), '-std:', np.std(np.array(jaccard)))
    print('-average mAP:', np.mean(np.array(mAPs)), '-std:', np.std(np.array(mAPs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
