# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import argparse
import torch 
import json

import numpy as np 
import matplotlib.pyplot as plt

from texttable import Texttable
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
                        MeanSquaredError, MeanAbsoluteError, JaccardIndex


from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.metric import Metric

try: 
    from .cc import NormalizedCrossCorrelation
    from .nmi import MutualInformation
    from .mpa import MeanPixelAccuracy
except Exception as e:
    from cc import NormalizedCrossCorrelation
    from nmi import MutualInformation
    from mpa import MeanPixelAccuracy


class Metrics():

    def __init__(self, device, metrics=['MSE', 'MAE', 'PSNR', 'CC', 'IoU', 'MPA']): 
        
        # Per-pixel metrics
        self.mse = MeanSquaredError(reduction='elementwise_mean').to(device)
        self.mae = MeanAbsoluteError(reduction='elementwise_mean').to(device)

        # Perceptual metrics
        self.psnr = PeakSignalNoiseRatio(reduction='elementwise_mean').to(device)
        self.ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
        self.jaccard =  MulticlassJaccardIndex(num_classes=2).to(device) 
        self.cc = NormalizedCrossCorrelation(reduction='mean').to(device)
        self.nmi = MutualInformation().to(device)
        self.mpa = MeanPixelAccuracy().to(device)
        
        self.metric_fns = {
            'MSE': self.mse, 
            'MAE': self.mae, 
            'SSIM': self.ssim,
            'PSNR': self.psnr,
            'IoU': self.jaccard,
            'CC': self.cc,
            'NMI': self.nmi,
            'MPA': self.mpa,
        }

        self.device = device
        self.metrics = metrics

    def forward(self, preds, target):
        metrics = dict()
        for name in self.metrics:
            fn = self.metric_fns[name]
            scores = fn(preds, target)   
            metrics[name] = torch.mean(scores)
                
        return metrics


def preprocess_iou(preds, target, device):
    target[target <= 0] = 0
    preds[preds <= 0] = 0

    classes = torch.unique(target)
    
    y_pred_mapped = []
    for _, e in enumerate(preds.flatten()):
        y_pred_mapped.append(classes[abs(classes-e).argmin()])

    preds = torch.tensor(y_pred_mapped, dtype=torch.float32, device=device).reshape(target.shape)

    return preds, target


def tabulate_runs(metrics, run_time=None, save_path=None):
    stats = dict()
    table = Texttable()

    keys = list(metrics[0].keys())

    for key in keys:
        values = []
        for metric in metrics: 
            values.append(metric[key].cpu().detach().numpy())

        stats[key] = (np.mean(values), np.std(values))

    if run_time:
        stats["run_time"] = (np.mean(run_time), np.std(run_time))
        keys.append("run_time")

    table.set_cols_align(["c"]*len(keys))
    table.add_rows([keys, [f"{stats[key][0]} \u00B1 {stats[key][1]}" for key in keys]])
    
    if save_path: 
        with open(save_path, 'w') as fp:
            json.dump(str(stats), fp) 

    return stats, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', type=str, help="Path to image1.")
    parser.add_argument('--image2', type=str, help="Path to image1.")

    args = parser.parse_args()
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image1 = torch.tensor(np.load(args.image1),  device=device, dtype=torch.float32) 
    image2 = torch.tensor(np.load(args.image2),  device=device, dtype=torch.float32)

    image1_binary = torch.clone(image1)
    image2_binary = torch.clone(image2)

    # convert to binary images with two classes 
    image1_binary[image1 != 1] = 1
    image1_binary[image1 == 1] = 0
    image2_binary[image2 == 1] = 0
    image2_binary[image2 != 1] = 1

    # plt.imshow(image1_binary)
    # plt.savefig("image1.png")
    # plt.imshow(image2_binary)
    # plt.savefig("image2.png")

    # reshape to [BxCXWxH]
    image1_binary = image1_binary.reshape((1, 1, image1_binary.shape[0], image1_binary.shape[1]))
    image2_binary = image2_binary.reshape((1, 1, image2_binary.shape[0], image2_binary.shape[1]))

    metrics = Metrics(device=device)

    metrics_val = metrics.forward(image1_binary, image2_binary)
    ideal_val = metrics.forward(image1_binary, image1_binary)

    stats, table = tabulate_runs([metrics_val], [0], "output.json")

    print(table.draw())


if __name__ == "__main__":
    main()