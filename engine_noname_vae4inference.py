import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation import get_pr_re_acc_jacc, get_map

def evaluate(model, data_loader, device, output_dir=None):
    model.eval()
    accuracy, precision, recall, mAPs, jaccard = [], [], [], [], []
    plt.figure(figsize=(50, 10))
    with torch.no_grad():
        print('------start inference------')
        for index, (samples, targets) in enumerate(data_loader):
            if (targets['id'][0] == 'video14') or (targets['id'][0] == 'video12'):
                continue

            print(index,'--', len(data_loader), '--', targets['id'][0])
            samples = samples.to(device)
            pred_phase, pred_tool, _ = model(samples, training=False)#
            pred_phase_digits = F.softmax(pred_phase, dim=1)
            pred_phase_label = torch.argmax(pred_phase_digits, dim=1).detach().cpu().numpy()
            gt_phase_label = targets['phase_label'].numpy()[0]

            pred_tool_label = F.sigmoid(pred_tool).detach().cpu().numpy().transpose()
            gt_tool_label = targets['tool_label'].numpy()[0].transpose()

            acc, pr, re, jac = get_pr_re_acc_jacc(input=pred_phase_label, label=gt_phase_label)
            map = get_map(input=pred_tool_label, label=gt_tool_label)

            accuracy.append(acc)
            precision.append(pr)
            recall.append(re)
            mAPs.append(map)
            jaccard.append(jac)

            print('acc:',acc,'pr:',pr,'re:',re, 'jaccard:', jac,'map:', map)
 
        print('-average accuracy:', np.mean(np.array(accuracy)), '-std:', np.std(np.array(accuracy)))
        print('-average precision:', np.mean(np.array(precision)), '-std:', np.std(np.array(precision)))
        print('-average recall:', np.mean(np.array(recall)), '-std:', np.std(np.array(recall)))
        print('-average jaccard:', np.mean(np.array(jaccard)), '-std:', np.std(np.array(jaccard)))
        print('-average mAP:', np.mean(np.array(mAPs)), '-std:', np.std(np.array(mAPs)))
    return

