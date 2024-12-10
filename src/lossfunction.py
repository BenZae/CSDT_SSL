import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from skimage.measure import label, regionprops
import cv2

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_preds, teacher_preds):
        return torch.mean((student_preds - teacher_preds) ** 2)

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5], smooth=1e-5, sigmoid=False, squared_pred=False):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.smooth = smooth
        self.sigmoid = sigmoid
        self.squared_pred = squared_pred

    def forward(self, logit, truth):
        if self.sigmoid:
            logit = torch.sigmoid(logit)

        if self.squared_pred:
            logit = logit ** 2
            truth = truth ** 2

        assert logit.shape == truth.shape, "预测值和真实值的形状必须相同"

        logit_flat = logit.view(logit.size(0), -1)
        truth_flat = truth.view(truth.size(0), -1)
        w = truth_flat.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        p = w * logit_flat
        t = w * truth_flat

        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)

        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        loss = dice.mean()

        return loss

class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.4, 0.6], sigmoid=False):
        super(WeightedBCE, self).__init__()
        self.weights = weights
        self.sigmoid = sigmoid

    def forward(self, logit, truth):
        if self.sigmoid:
            logit = torch.sigmoid(logit)

        logit = logit.view(-1)
        truth = truth.view(-1)
        assert logit.shape == truth.shape, "预测值和真实值的形状必须相同"

        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()

        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

        return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1, sigmoid=False):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5], sigmoid=sigmoid)
        self.dice_loss = DiceLoss()
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-5, size_average=True, sigmoid=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        self.sigmoid = sigmoid

    def forward(self, inputs, targets):
        if self.sigmoid:
            inputs = torch.sigmoid(inputs)

        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()

class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=1, dice_weight=1, alpha=0.8, gamma=2.0, smooth=1e-5, sigmoid=True, squared_pred=False):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, sigmoid=sigmoid)
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5], smooth=smooth, sigmoid=sigmoid, squared_pred=squared_pred)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal_dice_loss = self.focal_weight * focal + self.dice_weight * dice
        return focal_dice_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, sigmoid=False, squared_pred=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
        self.squared_pred = squared_pred

    def forward(self, logit, truth):
        if self.sigmoid:
            logit = torch.sigmoid(logit)

        if self.squared_pred:
            logit = logit ** 2
            truth = truth ** 2

        assert logit.shape == truth.shape, "预测值和真实值的形状必须相同"

        logit_flat = logit.view(logit.size(0), -1)
        truth_flat = truth.view(truth.size(0), -1)

        intersection = (logit_flat * truth_flat).sum(-1)
        union = logit_flat.sum(-1) + truth_flat.sum(-1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice

        return loss.mean()