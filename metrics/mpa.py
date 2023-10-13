
"""
    Calculate Mean Pixel Accuracy
    Source: https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch/blob/master/metric.py
"""

import torch 

from torch import nn
import numpy as np


class MeanPixelAccuracy(nn.Module):

    def __init__(self, eps=1e-5):
        super(MeanPixelAccuracy, self).__init__()
        self.eps = eps  

    
    def forward(self, preds, target):
        preds = preds.view(-1, )
        target = target.view(-1, ).float()

        tp = torch.sum(preds * target)  # TP
        fp = torch.sum(preds * (1 - target))  # FP
        fn = torch.sum((1 - preds) * target)  # FN
        tn = torch.sum((1 - preds) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)

        return pixel_acc