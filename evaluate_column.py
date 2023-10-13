# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os 
import time
import math
import argparse 
import cv2 
import numpy as np
import torch

from torch.utils.data import DataLoader

from data.dataset import Dataset
from data.plot import draw_grid, draw_confocal_grid, sweep_frame

from model.model import Generator, ResidualGenerator
from model.train import test, smooth_predictions
from model.loss import get_loss 

from scipy.ndimage import gaussian_filter, median_filter

from data.utils import read_yaml, resize_cfg
from data.confocal import read_confocal, build_depth_image_2, conf_image_size, preprocess, plot_conf_images, plot_confocal
from data.minerva import read_ect, get_ect_data

from config import combine_cfgs
from utils import init_torch_seeds, load_checkpoint
from experiments.tree_generator import TreeGenerator
from metrics.metrics import Metrics, tabulate_runs


def post_process(pred):
    # 2. Remove small dots from image 
    kernel = np.ones((10, 10),np.uint8)
    img_dilation = cv2.dilate(pred, kernel, iterations=1)
    img_dilation = median_filter(img_dilation, size=10)
    kernel = np.ones((12, 12),np.uint8)
    erosion = cv2.erode(img_dilation, kernel, iterations = 1)
    # fill 
    return erosion 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to training configuration.", required=True)
    parser.add_argument('--model', type=str, help="Path to the trained model", required=False)
    parser.add_argument('--ect', type=str, help="Path to ECT Data", required=True)
    parser.add_argument('--confocal', type=str, help="Path to Confocal Data", required=True)
    parser.add_argument('--ect_cfg', type=str, help="Path to ECT Data", required=True)
    parser.add_argument('--confocal_cfg', type=str, help="Path to Confocal Data", required=True)
    parser.add_argument('--slice_col', type=int, help="Column Slice to Predict", required=True)
    parser.add_argument('--batch_size', type=int, help="Batch Size", required=False, default=1)
    parser.add_argument('--output_dir', type=str, help="Batch Size", required=False, default="logs/column2")

    args = parser.parse_args()
    
    model_path = args.model 
    ect_file = args.ect 
    confocal_file = args.confocal
    ect_cfg_file = args.ect_cfg
    confocal_cfg_file = args.confocal_cfg
    slice_col = args.slice_col
    output_dir = args.output_dir 

    config = combine_cfgs(args.config)

    seed = config.SEED 
    exp_name = config.NAME 
    num_measurements = config.DATASET.NUM_MEASUREMENTS 
    head_activation = config.MODEL.HEAD_ACTIVATION
    hidden_activation = config.MODEL.HIDDEN_ACTIVATION 
    loss = config.SOLVER.LOSS
    model_type = config.MODEL.TYPE

    if args.batch_size: 
        batch_size = args.batch_size 
    else: 
        batch_size = config.DATASET.BATCH_SIZE
   
    # save_path = os.path.join('experiments', exp_name)
    # output_dir = os.path.join(save_path, "eval")

    if model_path is None: 
        model_path = os.path.join(save_path, 'best_model.pth')
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_tree = TreeGenerator(root_dir=output_dir)
    output_tree.generate()
    
    # Prepare model and load parameters
    if model_type == 'Vanilla-Decoder':
        model = Generator(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)
    else: 
        model = ResidualGenerator(input_dim=num_measurements, head_activation=head_activation, hidden_activation=hidden_activation)

    model.load_state_dict(torch.load(model_path)['state_dict']) 
    model = model.to(device)


    # Read ECT & Confocal Datasets
    ect_cfg = read_yaml(ect_cfg_file)
    conf_cfg = read_yaml(confocal_cfg_file) 
    
    conf_img_stack, conf_image, conf_maxZ = read_confocal(confocal_file, conf_cfg)
    print("Confocal shape: ", conf_image.shape)

    ect_images, row_offsets, col_offsets = read_ect(ect_file, ect_cfg, output_dir)
    print("Row Offsets: ", row_offsets)
    
    num_rows = ect_images[0].shape[0] - ect_cfg.ROW_OFFSET
    stride = ect_cfg.ROW_OFFSET
    
    predictions = torch.tensor([], device=device)
    pred_processed = torch.tensor([], device=device)
    ground_truth = torch.tensor([], device=device, dtype=torch.float32)
    
    min = -0.13654564083172416
    # np.min(ect_images)
    max = 1.223320999768493
    # np.max(ect_images)

    print(min, max)

    for i in range(0, num_rows, stride):
        row_range = [i, i+ect_cfg.ROW_OFFSET]

        # get corresponding cross sectional image from  confocal 
        if not conf_cfg.RESIZE_STACK: 
            confocal_column = math.ceil((slice_col * 10) / conf_cfg.PIXEL_SIZE_XY)
            conf_step = math.ceil((i*10)/conf_cfg.PIXEL_SIZE_XY) + 1
            conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]
        else:
            confocal_column = slice_col
            conf_step = i*10
            conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]
            
        # quit if the confocal range is above the confocal image
        if conf_row_range[1] > conf_image.shape[0]: 
            break 
                
        minerva_data = get_ect_data(ect_images, row_offsets, ect_cfg.MAX_ROW_OFFSET, ect_cfg.MIN_ROW_OFFSET, slice_col, row_range, [slice_col, slice_col+ect_cfg.COL_OFFSET], ect_cfg, output_dir) 

        scaled_data = minerva_data
        scaled_data[:, 2] = minerva_data[:, 2]* 1e15 * 0.1
        
        # min max scaling for the data 
        vb = scaled_data[:, 2]

        vb = (vb - min) / (max - min)

        vb = torch.tensor(vb, device=device)
        vb = vb.view((1, vb.shape[0], 1, 1))

        predicted_perm = model(vb.float())
        pred_perm_smoothed, _ = smooth_predictions(predicted_perm, torch.tensor([]), config.MODEL.HEAD_ACTIVATION, config.DATASET.POS_VALUE, config.DATASET.NEG_VALUE)
        # pred_perm_smoothed = predicted_perm
        
        predictions = torch.cat((predictions, pred_perm_smoothed), 3)
        draw_grid(pred_perm_smoothed[0][0].cpu().detach().numpy(), "predicted_perm", "", "", os.path.join(output_tree.pred_path, f"pred_{i}.png"))

        pred_processed_pred = post_process(pred_perm_smoothed[0][0].cpu().detach().numpy()).reshape(1, 1, pred_perm_smoothed.shape[2], pred_perm_smoothed.shape[3])
        pred_processed = torch.cat((pred_processed, torch.tensor(pred_processed_pred, device=device)), 3)

        cross_section = build_depth_image_2(conf_img_stack, conf_cfg, conf_row_range, confocal_column)

        save_path = os.path.join(output_tree.true_path, f"ground_truth_{i}.png")
        # cross_section_processed = preprocess(cross_section, save_path)
        # cross_section_processed[cross_section_processed == 255] = conf_cfg.BACKGROUND_PERM 
        # cross_section_processed[cross_section_processed == 0] = conf_cfg.FOREGROUND_PERM 
        cross_section_processed = np.array(cross_section, dtype=np.float32)
        
        save_path = os.path.join(output_tree.true_path, f"frame_{i}.png")
        sweep_frame(slice_col, confocal_column, row_range, conf_row_range, ect_images[row_offsets.index(-1)], conf_image,  minerva_data[:, 2], scaled_data[:, 2], cross_section_processed, cross_section_processed, cross_section, save_path)

        # normalize cross section with the pixel size 
        dsize = (int(cross_section_processed.shape[1]*conf_cfg.PIXEL_SIZE_XY), int(cross_section_processed.shape[0]*conf_cfg.PIXEL_SIZE_Z))
        cross_section_resized = cv2.resize(cross_section_processed, dsize=dsize, interpolation=cv2.INTER_CUBIC) 
        
        # resize to desired size 
        rows, cols = conf_cfg.DSIZE
        image = cross_section_resized[0:rows, 0:cols]
        
        save_path = os.path.join(output_tree.true_path,  f"{i}_cross_section_{slice_col}.png")
        plot_confocal(image, "Cross-sectional Image", "x", "Depth (z) in microns", save_path) 

        cross_section_processed = torch.tensor(cross_section_processed, device=device)        
        ground_truth = torch.cat((ground_truth, cross_section_processed), 1)


    # flatten the predictions
    predictions = predictions[0][0].cpu().detach().numpy()
    pred_processed = pred_processed[0][0].cpu().detach().numpy()

    ground_truth = ground_truth[0:100, :].cpu().detach().numpy()
    
    predictions = cv2.resize(predictions,None, fx=1, fy=1)
    ground_truth = cv2.resize(ground_truth, None, fx=1, fy=1)[:, 50:]   # discard the first 50 microns
    
    # pred_processed = post_process(predictions)

    draw_grid(predictions,  f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"pred_{slice_col}.png"),  figsize=(26, 13), cmap='viridis', colorbar=False, scale_bar=True, ticks=False, aspect_ratio=5.5, font_size=30)
    draw_grid(pred_processed,  f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"pred_processed_{slice_col}.png"),  figsize=(26, 13), cmap='viridis', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=5.5, font_size=30)
    draw_grid(predictions,  f"ECT Prediction", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"pred_{slice_col}.pdf"),  figsize=(26, 13), cmap='viridis', colorbar=False, scale_bar=True, ticks=False,  aspect_ratio=5.5, font_size=30, format="pdf")
    draw_grid(ground_truth, f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"truth_{slice_col}.png"),  figsize=(26, 13), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=5.5, font_size=30)
    draw_grid(ground_truth, f"Confocal Microscopy", "Row(y)", "Depth(z)", os.path.join(output_tree.root_dir, f"truth_{slice_col}.pdf"),  figsize=(26, 13), cmap='Reds', colorbar=False, scale_bar=True,  ticks=False, aspect_ratio=5.5, font_size=30, format="pdf")

    # predictions_flattened = torch.flatten(predictions, 0, 1)
    # ground_truth_flattened = torch.flatten(ground_truth, 0, 1)
    
    # print(predictions_flattened.shape)
    # print(ground_truth_flattened.shape)
    
    # metrics = Metrics(device=device)
    # metrics = metrics.forward(predictions, ground_truth)
    # print(metrics)

    # save_path = os.path.join(output_dir, "stats.json")
    # stats, table = tabulate_runs([metrics], run_time, save_path)
    # print(table.draw())


if __name__ == "__main__":
    main()