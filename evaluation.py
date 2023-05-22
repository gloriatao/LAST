import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.metrics import average_precision_score #(true_labels, predicted_probs)

def get_pr_re_acc_jacc(input, label):
    precision, recall, jaccard = [], [], []
    for l in np.unique(label):
        tmp_gt = np.zeros(label.shape)
        tmp_gt[label==l] = 1
        tmp_pred = np.zeros(input.shape)
        tmp_pred[input==l] = 1
        sum = (tmp_pred * tmp_gt).sum()
        pr = sum/tmp_pred.sum()
        re = sum/tmp_gt.sum()

        jacd = jaccard_score(tmp_gt, tmp_pred)        
        if jacd == 0.0:
            # non-existing phase class, no positive class found in y_true
            continue

        precision.append(pr)
        recall.append(re)
        jaccard.append(jacd)

    precision, recall, jaccard = np.mean(np.array(precision)), np.mean(np.array(recall)), np.mean(np.array(jaccard))
    accuracy = accuracy_score(label, input)
    return accuracy, precision, recall, jaccard


def get_map(input, label):
    num_class = label.shape[0]
    mAP = []
    for t in range(num_class):
        pred = input[t,:]
        gt = label[t,:]
        ap = average_precision_score(gt, pred)
        if ap == 0.0:
            # non-existing tool class, no positive class found in y_true
            continue

        mAP.append(ap)
    mAP = np.mean(np.array(mAP))
    return mAP

