import torch
import numpy as np

def mean_iou(pred, target, num_classes):
    ious = []
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    
    return np.nanmean(ious)