# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import argparse 
import sys
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import matplotlib.pyplot as plt


def get_loss(name, **kwargs): 
    loss_fn = None

    if name == 'IoU':
        loss_fn = IoULoss()
  
    if name == 'FL':
        gamma = kwargs.get('gamma', 2)
        alpha = kwargs.get('alpha', 0.5)
        loss_fn = FocalLoss(gamma, alpha)

    if name == 'L1Smooth':
        loss_fn =  nn.SmoothL1Loss()
        
    if name == 'FL+L1Smooth':
        gamma = kwargs.get('gamma', 2)
        alpha = kwargs.get('alpha', 0.5)
        weights = kwargs.get('weights', [1, 1])
        trainable_weights = kwargs.get('trainable_weights', False)
        loss_fn = FocalL1Loss(gamma, alpha, weights, trainable_weights)
    
 
    if name == 'FL+Dice+L1Smooth':
        gamma = kwargs.get('gamma', 2)
        alpha = kwargs.get('alpha', 0.5)
        weights = kwargs.get('weights', [1, 1, 1])
        trainable_weights = kwargs.get('trainable_weights', False)
        loss_fn = FocalDiceL1Loss(gamma, alpha, weights, trainable_weights=trainable_weights)

    if name == 'FL+Dice+L2':
        gamma = kwargs.get('gamma', 2)
        alpha = kwargs.get('alpha', 0.5)
        weights = kwargs.get('weights', [1, 1, 1])
        trainable_weights = kwargs.get('trainable_weights', False)
        loss_fn = FocalDiceL2Loss(gamma, alpha, weights, trainable_weights=trainable_weights)

  
    if loss_fn is None:
        print(f"Invalid Loss Name {name}") 
        sys.exit(1)

    return loss_fn


class FocalLoss(nn.Module):
    """
        Implementation for Focal Loss function 
        https://amaarora.github.io/2020/06/29/FocalLoss.html
    """

    def __init__(self, weight=None, gamma=2, alpha=0.5, reduction='none', device='cuda'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.tensor([alpha, 1-alpha], device=device)
        self.reduction = reduction
        
    
    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()

        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.alpha = self.alpha.to(device)
        at = self.alpha.gather(0, targets.data.view(-1))
        at = at.reshape(inputs.shape)
        
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss

        return F_loss.mean()

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
    
    

class FocalL1Loss(nn.Module):

    def __init__(self, gamma=2, alpha=0.5, weights=[1, 0.5], trainable_weights=False, reduction='none'): 
        super(FocalL1Loss, self).__init__()        

        if trainable_weights:
            self.awl = AutomaticWeightedLoss(3)

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.l1_smooth = nn.SmoothL1Loss()
        self.trainable_weights = trainable_weights

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        l1_loss = self.l1_smooth(inputs, targets)

        if self.trainable_weights: 
            loss = self.awl(focal_loss, l1_loss)
        else:
            loss = self.weights[0]*focal_loss + self.weights[1]*l1_loss 
            
        return loss 


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
                
        #flatten label and prediction tensorss
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return -1*IoU


class FocalDiceL1Loss(nn.Module):

    def __init__(self,  gamma, alpha, weights=[1, 1, 1], trainable_weights=False, smooth=1):
        super(FocalDiceL1Loss, self).__init__()

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.l1_smooth = nn.SmoothL1Loss()
        self.dice_loss = DiceLoss() 

        if trainable_weights: 
            self.awl = AutomaticWeightedLoss(3)

        self.trainable_weights = trainable_weights
        self.weights = weights 
        
    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)
        l1_loss = self.l1_smooth(pred, target)
        dice_loss = self.dice_loss(pred, target)

        if self.trainable_weights: 
            loss = self.awl(focal_loss, dice_loss, l1_loss)
        else: 
            loss = self.weights[0] * focal_loss  + self.weights[1] * dice_loss + self.weights[2] * l1_loss

        return loss


class FocalDiceL2Loss(nn.Module):

    def __init__(self,  gamma, alpha, weights=[1, 1, 1], trainable_weights=False, smooth=1):
        super(FocalDiceL2Loss, self).__init__()

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.l2_loss = nn.MSELoss()
        self.dice_loss = DiceLoss() 

        if trainable_weights: 
            self.awl = AutomaticWeightedLoss(3)

        self.trainable_weights = trainable_weights
        self.weights = weights 
        
    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)
        l1_loss = self.l2_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        if self.trainable_weights: 
            loss = self.awl(focal_loss, dice_loss, l1_loss)
        else: 
            loss = self.weights[0] * focal_loss  + self.weights[1] * dice_loss + self.weights[2] * l1_loss

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth 
    
    def forward(self, y_pred, y_true):
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = (y_pred_f * y_true_f).sum()

        dice_coef = (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)

        dice_loss = 1 - dice_coef
        
        return dice_loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', type=str, help="Path to image1.")
    parser.add_argument('--image2', type=str, help="Path to image1.")

    args = parser.parse_args()
    
    if args.image1: 
        image1_path = args.image1
    else: 
        image1_path = "../data/synthetic/data/dataset3/x/perm_xy/0__beads_2.npy"
    
    if args.image2: 
        image2_path = args.image2 
    else: 
        image2_path = "../data/synthetic/data/dataset3/x/perm_xy/1__beads_2.npy"
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image1 = torch.tensor(np.load(image1_path), dtype=torch.float32) 
    image2 = torch.tensor(np.load(image2_path), dtype=torch.float32)

    image1_binary = torch.clone(image1)
    image2_binary = torch.clone(image2)

    # convert to binary images with two classes 
    image1_binary[image1 != 1] = 1
    image1_binary[image1 == 1] = 0
    image2_binary[image2 == 1] = 0.2
    image2_binary[image2 != 1] = 0.8
    
    plt.imshow(image1_binary)
    plt.colorbar()
    plt.savefig("image1.png")
    plt.imshow(image2_binary)
    plt.savefig("image2.png")

    # reshape to [BxCXWxH]
    image1_binary = image1_binary.reshape((1, 1, image1_binary.shape[0], image1_binary.shape[1]))
    image2_binary = image2_binary.reshape((1, 1, image2_binary.shape[0], image2_binary.shape[1]))
    
        
    image1_binary = image1_binary.to(device)
    image2_binary = image2_binary.to(device)

   


if __name__ == "__main__":
    main()