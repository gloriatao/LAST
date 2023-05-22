import datetime, os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from dataset.load_val_vfe_savepickle import load_valset
from engine_twoheadst import evaluate_save_pickle
from models import VFE
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # Model parameters
    parser.add_argument('--data_path', type=str, default='/media/cygzz/data/rtao/data/cholec/cholec80/npys')
    parser.add_argument('--output_dir', default='validation_npys', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0], help='device to use for training / testing')
    parser.add_argument('--seed', default=130, type=int)
    parser.add_argument('--resume', default='weights/stage1-light.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def main(args):
    print(os.environ)
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
    print('number of params:', n_parameters)

    dataset_val = load_valset(args.data_path)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])


    print("Start training")
    start_time = time.time()

    # train loop--------------------------------------
    iteration = args.start_iteration
    print('start iteration:', iteration)
    evaluate_save_pickle(model, data_loader_val, device, args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
