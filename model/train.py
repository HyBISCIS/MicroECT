# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 


import os
import sys 
import time 

import torch 
torch.cuda.empty_cache()

import numpy as np
from torch.autograd import Variable

from data.plot import draw_grid, draw_confocal_grid, draw_line


def train_one_epoch(generator, optimizer, loss_fn, train_loader, val_loader, epoch, device, noise=False, noise_stdv=0.02):
    train_losses = 0
    val_losses = 0

    for x, y in train_loader: 
        perm  = x["perm_xy"]
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        optimizer.zero_grad()

        generator.train(True)

        # perturb input with gaussian noise
        if noise:
            vb =  vb + Variable(torch.randn(vb.shape, device=device) * noise_stdv)
    
        input_g = vb.view(vb.shape[0], vb.shape[1], 1, 1)
        predicted_perm = generator(input_g.float())
        predicted_perm = predicted_perm.to(device)
        
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                        for p in generator.parameters())

        loss = loss_fn(predicted_perm, perm)

        loss.backward()
        optimizer.step()

        train_losses += loss.item()

    generator.train(False)

    for x, y in val_loader:
        perm  = x["perm_xy"]
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        # perturb input with gaussian noise
        if noise:
            vb = vb + Variable(torch.randn(vb.shape, device=device) * noise_stdv)
            
        input_g = vb.view(vb.shape[0], vb.shape[1], 1, 1)
        predicted_perm = generator(input_g.float())

        loss = loss_fn(predicted_perm, perm) 
        val_losses += loss.item() 
   
    train_avg_loss = train_losses / len(train_loader)
    val_avg_loss = val_losses / len(val_loader)

    print('Epoch: %0.2f | Training Loss: %.6f | Validation Loss: %0.6f'  % (epoch, train_avg_loss, val_avg_loss), flush=True)
    # print(loss_fn.awl.params, flush=True)

    return train_avg_loss, val_avg_loss 


def test(generator, loss_fn, test_loader, config, output_tree, device):
    test_loss = 0
    predictions = torch.tensor([])
    predictions = predictions.to(device)

    ground_truth = torch.tensor([])
    ground_truth = ground_truth.to(device)

    generator.train(False)

    for i, (x, y) in enumerate(test_loader):
        perm  = x["perm_xy"].float()
        vb = y["v_b"]

        perm = perm.to(device)
        vb = vb.to(device)

        input_g = vb.view(vb.shape[0], vb.shape[1], 1, 1)

        st = time.time()
        predicted_perm = generator(input_g.float())
        end = time.time() - st 

        # smooth predictions
        pred_perm_smoothed, ground_truth_smoothed = smooth_predictions(predicted_perm, perm, config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)

        predictions = torch.cat((predictions, pred_perm_smoothed))
        ground_truth = torch.cat((ground_truth, ground_truth_smoothed))
        
        loss = loss_fn(predicted_perm, perm) 
        test_loss += loss.item()  

        for j, pred in enumerate(pred_perm_smoothed): 
            pred_perm = pred
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", "Row (200\u03bcm)", "Depth (100\u03bcm)",  os.path.join(output_tree.pred_path, f"pred_{i}_{j}.png"), font_size=24, figsize=(6, 6), cmap='viridis', colorbar=False)
            draw_grid(pred_perm[0].cpu().detach().numpy(), "ECT Prediction", "Row (y)", "Depth (z)",  os.path.join(output_tree.pred_path, f"pred_scale_bar_{i}_{j}.png"), font_size=24, figsize=(6, 6), cmap='viridis', ticks=False, scale_bar=True, colorbar=False)

            draw_confocal_grid(ground_truth_smoothed[j][0].cpu().detach().numpy(), "Ground Truth", "Row (200\u03bcm)", "Depth (100\u03bcm)",  os.path.join(output_tree.true_path, f"truth_{i}_{j}.png"), font_size=24, figsize=(6, 6), cmap='Reds', colorbar=False)
            draw_confocal_grid(ground_truth_smoothed[j][0].cpu().detach().numpy(), "Ground Truth", "Row (y)", "Depth (z)",  os.path.join(output_tree.true_path, f"truth_scale_bar_{i}_{j}.png"), font_size=24, figsize=(6, 6), cmap='Reds', ticks=False, scale_bar=True, colorbar=False)

            # break 
        

    test_avg_loss = test_loss / len(test_loader)
    
    print("Testing Loss: %0.6f" % (test_avg_loss))

    return test_avg_loss, predictions, ground_truth


def smooth_predictions(predicted_perm, ground_truth, activation, pos_value, neg_value):
    pred_perm = torch.clone(predicted_perm)
    truth_perm = torch.clone(ground_truth)

    if activation == 'Tanh':
        pred_perm[predicted_perm < 0] = 1 # pos_value
        pred_perm[predicted_perm >= 0] = 0 # neg_value
        truth_perm[ground_truth < 0] = 1 
        truth_perm[ground_truth >= 0] = 0 
        
        
    if activation == 'Sigmoid':
        pred_perm[predicted_perm >= 0.45] = pos_value
        pred_perm[predicted_perm < 0.45] = neg_value


    return pred_perm, truth_perm