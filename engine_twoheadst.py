import math
import os
import sys
import numpy as np
import torch, pickle
import util.misc as utils
import torch.nn.functional as F
from pathlib import Path

def train_one_epoch(model, data_loader, optimizer, device, epoch, log_path, output_dir, lr_scheduler, steps, max_norm,
                    data_loader_val, val_log_dir, avg_acc_benchmark, iteration):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc_phase', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    lsses = []
    label_weights = [
        1.6411019141231247,
        0.19090963801041133,
        1.0,
        0.2502662616859295,
        1.9176363911137977,
        0.9840248158200853,
        2.174635818337618,
    ]
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        model.train()
        samples = samples.to(device)
        pred_phase, pred_tool, _ = model(samples)
        # loss
        loss_phase = torch.nn.CrossEntropyLoss(weight = torch.tensor(label_weights).cuda())(pred_phase, targets['phase_label'].cuda())  #weight = torch.tensor(label_weights).cuda()
        loss_tool = torch.nn.BCELoss()(F.sigmoid(pred_tool), targets['tool_label'].float().cuda())

        loss_dict = {'phase':loss_phase, 'tool':loss_tool}
        weight_dict = {'phase':1, 'tool':1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = losses.item()

        # acc
        pred_phase_digits = F.softmax(pred_phase, dim=1)
        pred_p = torch.argmax(pred_phase_digits,dim=-1)
        correct = pred_p.eq(targets['phase_label'].cuda()).sum().float()
        total = float(len(pred_p))
        acc_p = correct/total

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        lr_scheduler.step_update(iteration)

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(acc_phase=acc_p)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # log
        lss = loss_dict.copy()
        for i, k in enumerate(lss):
            lss[k] = lss[k].detach().cpu().numpy().tolist()
        lss['iteration'] = iteration
        lss['epoch'] = epoch
        lsses.append(lss)
        with open(os.path.join(log_path, str(iteration)+'.pickle'), 'wb') as file:
            pickle.dump(lsses, file)
        file.close()

        # ex
        if (iteration%150 == 0) and (iteration != 0):
            checkpoint_paths = [output_dir / 'checkpoint_twoheadst.pth']
            pred_acc = evaluate(model, data_loader_val, device, iteration, val_log_dir)
            if pred_acc >= avg_acc_benchmark:
                checkpoint_paths = [output_dir / f'checkpoint_twoheadst_{pred_acc:04}.pth']
                print('saving best acc@', pred_acc, 'iteration:', iteration)
                avg_acc_benchmark = pred_acc

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'iteration': iteration,
                }, checkpoint_path)

        iteration+=1
        steps += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('avg_acc_benchmark:',avg_acc_benchmark)
    return avg_acc_benchmark, iteration

def evaluate(model, data_loader, device, iteration, output_dir):
    label_weights = [
        1.6411019141231247,
        0.19090963801041133,
        1.0,
        0.2502662616859295,
        1.9176363911137977,
        0.9840248158200853,
        2.174635818337618,
    ]
    model.eval()
    correct, total, loss_valid = 0,0,[]
    with torch.no_grad():
        print('start validation------')
        for index, (samples, targets) in enumerate(data_loader):
            print(index,'--', len(data_loader))
            samples = samples.to(device)
            pred_phase, pred_tool, _ = model(samples)
            # loss
            loss_phase = torch.nn.CrossEntropyLoss(weight = torch.tensor(label_weights).cuda())(pred_phase, targets['phase_label'].cuda())  #weight = torch.tensor(label_weights).cuda()
            loss_tool = torch.nn.BCELoss()(F.sigmoid(pred_tool), targets['tool_label'].float().cuda())

            loss_dict = {'phase':loss_phase, 'tool':loss_tool}
            loss_valid.append(loss_dict)

            # acc
            pred_phase_digits = F.softmax(pred_phase, dim=1)
            pred_p = torch.argmax(pred_phase_digits,dim=-1)
            total += float(len(pred_p))
            correct += pred_p.eq(targets['phase_label'].cuda()).sum().float()

        acc_p = correct/total


        results = {'loss_valid':loss_valid, 'acc_p':acc_p}
        print('avg_acc_phase--', acc_p)

        with open(os.path.join(output_dir, str(iteration) + '_valid.pickle'), 'wb') as file:
            pickle.dump(results, file)
        file.close()

    return acc_p

def evaluate_save_pickle(model, data_loader, device):
    outpath = '/media/cygzz/data/rtao/projects/cholec/validation_npys'
    model.eval()
    with torch.no_grad():
        print('start validation------')
        for index, (samples, targets) in enumerate(data_loader):
            print(index,'--', len(data_loader), targets)
            samples = samples.to(device).squeeze(0)
            _, _, xs = model(samples)
            # path---------------------------------------------
            feat = xs.detach().cpu().numpy()
            save_dir = Path(outpath+'/'+targets['id']['id'][0])
            print(save_dir)
            if os.path.isdir(save_dir) == False:
                os.mkdir(save_dir)

            print(os.path.join(save_dir, str(targets['id']['time'].numpy()[0])+'.npy'))
            np.save(os.path.join(save_dir, str(targets['id']['time'].numpy()[0])+'.npy'), feat)

    return

