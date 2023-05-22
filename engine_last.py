import math
import os
import sys
import torch, pickle
import util.misc as utils
import torch.nn.functional as F
import numpy as np

def f1(input, label):
    eps = torch.Tensor([1e-3]).cuda()
    loss = torch.zeros(1).cuda()
    if torch.sum(label) == 0:
        print('No mask in ground truth')
        return loss
    else:
        input = input.flatten()
        label = label.flatten()
        intersect = torch.dot(input, label)
        input_sum = torch.sum(torch.dot(input, input))
        label_sum = torch.sum(torch.dot(label, label))
        union = input_sum + label_sum
        loss = (2 * intersect + eps) / (union + eps)
    return loss

def train_one_epoch(model, data_loader, optimizer, device, epoch, log_path, output_dir, lr_scheduler, steps, max_norm,
                    data_loader_val, val_log_dir, avg_acc_benchmark, iteration, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc_phase', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('f1_t', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
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
        target_phase = targets['phase_label_m'].cuda().float()
        target_tool = targets['tool_label'].cuda().float()
        tar = torch.cat((target_phase, target_tool), dim=-1)
        pred_phase, pred_tool, kl_loss = model(samples, tar, training=True) #phase, tool, kl_loss, feat_phase, feat_tool
        # loss
        loss_phase = torch.nn.CrossEntropyLoss(weight = torch.tensor(label_weights).cuda())(pred_phase, targets['phase_label'].cuda().squeeze(0))  #weight = torch.tensor(label_weights).cuda()
        loss_tool = torch.nn.BCELoss()(F.sigmoid(pred_tool), targets['tool_label'].float().cuda().squeeze(0))

        loss_dict = {'phase': loss_phase, 'tool': loss_tool, 'kl': kl_loss}
        weight_dict = {'phase':0, 'tool':1,  'kl':100}  
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = losses.item()

        # acc
        pred_phase_digits = F.softmax(pred_phase, dim=1)
        pred_p = torch.argmax(pred_phase_digits, dim=-1)
        correct = pred_p.eq(targets['phase_label'].cuda()).sum().float()
        total = float(len(pred_p))
        acc_p = correct/total

        pred_t = pred_tool.clone()
        f1_t = f1(F.sigmoid(pred_t), targets['tool_label'].float().cuda())

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
        metric_logger.update(f1_t=f1_t)
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
        if (iteration%10 == 0) and (iteration != 0):
            if args.epochs_finetune_tool:
                checkpoint_paths = [output_dir / 'checkpoint_nonamest.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'iteration': iteration,
                    }, checkpoint_path)
            else:
                checkpoint_paths = [output_dir / 'checkpoint_nonamest.pth']
                acc, f1_t = evaluate(model, data_loader_val, device, iteration, val_log_dir)
                if acc >= avg_acc_benchmark:
                    checkpoint_paths = [output_dir / f'checkpoint_nonamest_{acc:04}.pth']
                    print('saving best acc_p @', acc, 'iteration:', iteration)
                    avg_acc_benchmark = acc

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

label_weights = [
    1.6411019141231247,
    0.19090963801041133,
    1.0,
    0.2502662616859295,
    1.9176363911137977,
    0.9840248158200853,
    2.174635818337618,
]

def evaluate(model, data_loader, device, iteration, output_dir):
    model.eval()
    loss_valid = []
    accs, f1_ts = [], []
    with torch.no_grad():
        print('start validation------')
        for index, (samples, targets) in enumerate(data_loader):
            print(index,'--', len(data_loader))
            samples = samples.to(device)
            target_phase = targets['phase_label_m'].cuda()
            pred_phase, pred_tool, kl_loss = model(samples, target_phase, training=False)
            # loss
            loss_phase = torch.nn.CrossEntropyLoss(weight=torch.tensor(label_weights).cuda())(pred_phase, targets['phase_label'].cuda().squeeze(0)) 
            loss_tool = torch.nn.BCEWithLogitsLoss()(pred_tool, targets['tool_label'].float().cuda().squeeze(0))

            loss_dict = {'phase': loss_phase, 'tool': loss_tool, 'kl': kl_loss}
            loss_valid.append(loss_dict)

            # acc
            pred_phase_digits = F.softmax(pred_phase, dim=1)
            pred_p = torch.argmax(pred_phase_digits,dim=-1)
            total = float(len(pred_p))
            correct = pred_p.eq(targets['phase_label'].cuda()).sum().float()
            acc_p = correct/total

            pred_t = pred_tool.clone()
            f1_t = f1(F.sigmoid(pred_t), targets['tool_label'].float().cuda())

            accs.append(acc_p.item())
            f1_ts.append(f1_t.item())

        results = {'loss_valid':loss_valid, 'acc_p':accs, 'f1_t':f1_ts}
        with open(os.path.join(output_dir, str(iteration) + '_valid.pickle'), 'wb') as file:
            pickle.dump(results, file)
        file.close()

        print(accs,'-' ,f1_ts)
        accs = np.mean(accs)
        f1_ts = np.mean(f1_ts)
        print('avg_acc_phase--', accs, 'avg_f1_tool--', f1_ts)

    return accs, f1_ts


