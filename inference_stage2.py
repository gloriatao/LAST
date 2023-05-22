import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from dataset.load_val_last import load_valset
from engine_noname_vae4inference import evaluate

from models import LAST

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int) # 500 one gpu
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, help="Path to the pretrained model.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    # dataset parameters
    parser.add_argument('--dataset_file', default='noname')
    parser.add_argument('--data_path', type=str, default='/media/cygzz/data/rtao/projects/cholec/validation_npys')
    parser.add_argument('--output_dir', default='pred', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0], help='device to use for training / testing')
    parser.add_argument('--seed', default=130, type=int)
    parser.add_argument('--resume', default='weights/stage2-light.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--band_widths', default=[100-1, 1000-1], type=int)
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

    model = LAST(num_phase_class=7, num_tool_class=7, band_width=args.band_widths)
    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model.to(device)

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)

    dataset_val = load_valset(file_paths=args.data_path, is_test=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    print("Start inference")

    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    pred_dir = Path(args.output_dir+'/stage2')
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    if os.path.isdir(pred_dir) == False:
        os.mkdir(pred_dir)

    evaluate(model, data_loader_val, device, pred_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
