from torch import nn
import torch
import torch.nn.functional as F
from crfseg import CRF


def dice_loss_multiclass(predict, target, weight=None, smooth=1e-5):
    N, C = predict.size()[:2]
    predict = predict.view(N, C, -1) # (N, C, *)
    target = target.view(N, 1, -1) # (N, 1, *)

    predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
    ## convert target(N, 1, *) into one hot vector (N, C, *)
    target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
    target_onehot.scatter_(1, target, 1)  # (N, C, *)

    intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
    union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
    ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
    dice_coef = (2 * intersection + smooth) / (union + smooth)  # (N, C)

    if weight is not None:
        if weight.type() != predict.type():
            weight = weight.type_as(predict)
            dice_coef = dice_coef * weight * C  # (N, C)
    dice_loss = 1 - torch.mean(dice_coef)  # 1

    return dice_loss

def dice_loss_binary(predict, target, weight=None, smooth=1e-5):
    logits_flat = predict
    targets_flat = target
    
    # Convert logits to binary predictions using threshold
    preds = (logits_flat >= 0).float()
    
    # Compute intersection and union (using sum of squares)
    intersection = (preds * targets_flat).sum()
    union = preds.pow(2).sum() + targets_flat.pow(2).sum()
    
    # Compute Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    
    # Compute Dice loss
    dice_loss = 1. - dice_coeff
    
    return dice_loss

crf = nn.Sequential(
        nn.Identity(),  # your NN
        CRF(n_spatial_dims=2)
    )

def crf_loss(predict):
    return (crf(predict) - predict).mean()


# class DiceLoss(nn.Module):
#     """
#     Args:
#         weight: An array of shape [C,]
#         predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
#         target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
#     Return:
#         diceloss
#     """
#     def __init__(self, weight=None):
#         super(DiceLoss, self).__init__()
#         self.weight = weight
#         if self.weight is not None:
#             self.weight = torch.Tensor(self.weight)
#             self.weight = self.weight / torch.sum(self.weight) # Normalized weight
#         self.smooth = 1e-5

#     def forward(self, predict, target):
#         return dice_loss(predict, target, self.weight, self.smooth)