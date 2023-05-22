import argparse
import datetime, os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from dataset.load_train_vfe import load_trainset
from dataset.load_val_vfe import load_valset
from engine_twoheadst import train_one_epoch
from models import VFE
from timm.scheduler.cosine_lr import CosineLRScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# number of params: 28,299,120
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--epochs_init', default=10, type=int)
    parser.add_argument('--epochs_finetune', default=10, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # Model parameters
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='/media/cygzz/data/rtao/data/cholec/cholec80/npys')
    parser.add_argument('--output_dir', default='stage1', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0,1], help='device to use for training / testing')
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

    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-4, weight_decay=0.05)

    dataset_train = load_trainset(args.data_path, args.epochs_finetune)
    dataset_val = load_valset(args.data_path)
    data_loader_train = DataLoader(dataset_train,batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size*2, drop_last=False, shuffle=False, num_workers=args.num_workers)

    num_steps = int(args.epochs_init * len(data_loader_train))
    warmup_steps = int(20 * len(data_loader_train))
    lr_scheduler = CosineLRScheduler(
                                    optimizer,
                                    t_initial=num_steps,
                                    lr_min=5e-6,
                                    warmup_lr_init=5e-7,
                                    warmup_t=warmup_steps,
                                    cycle_limit=1,
                                    t_in_epochs=False,
                                )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if  'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch']+1
            args.start_iteration = checkpoint['iteration']+1

    print("Start training")
    start_time = time.time()

    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    train_log_dir =  Path(args.output_dir+'/train_log')
    val_log_dir = Path(args.output_dir + '/val_log')
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    if os.path.isdir(train_log_dir) == False:
        os.mkdir(train_log_dir)
    if os.path.isdir(val_log_dir) == False:
        os.mkdir(val_log_dir)

    # train loop--------------------------------------
    avg_acc_benchmark, steps = 0.0, 0.0
    iteration = args.start_iteration
    print('start iteration:', iteration)
    for epoch in range(args.start_epoch, args.epochs_init):
        avg_acc_benchmark, iteration = train_one_epoch(model, data_loader_train, optimizer, device, epoch, train_log_dir, output_dir, lr_scheduler, steps, args.clip_max_norm,
                                data_loader_val, val_log_dir, avg_acc_benchmark, iteration)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
