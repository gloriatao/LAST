import numpy as np
import torch
import torch.nn.functional as F


def evaluate(model, data_loader, device):
    model.eval()
    pred_phase_label, gt_phase_label = [], []
    pred_tool_label, gt_tool_label = [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]]
    with torch.no_grad():
        for _, (samples, targets) in enumerate(data_loader):
            samples = samples.to(device)
            pred_phase, pred_tool, _ = model(samples)

            pred_phase_digits = F.softmax(pred_phase, dim=1)
            pred_phase_label += torch.argmax(pred_phase_digits, dim=1).detach().cpu().numpy().tolist()
            gt_phase_label += targets['phase_label'].numpy().tolist()

            pred_tool = torch.sigmoid(pred_tool)
            for i in range(pred_tool.shape[1]):
                pred_tool_label[i] += pred_tool[:,i].detach().cpu().numpy().tolist()
                gt_tool_label[i] += targets['tool_label'][:,i].numpy().tolist()

        pred_phase_label = np.array(pred_phase_label)
        gt_phase_label = np.array(gt_phase_label)

        pred_tool_label = np.array(pred_tool_label)
        gt_tool_label = np.array(gt_tool_label)

    return pred_phase_label, gt_phase_label, pred_tool_label, gt_tool_label
