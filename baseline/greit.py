# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import os 
import time 

import torch
import argparse
import numpy as np

from tqdm import tqdm
from pyeit.eit import greit
from pyeit.eit.fem import Forward
from torch.utils.data import DataLoader


from config import combine_cfgs
from data.dataset import Dataset
from data.mesh_params import MeshParams
from data.generate_data import create_mesh, create_ex_mat
from data.plot import interpolate_perm, draw_greit, draw_grid, draw_perm
from metrics.metrics import Metrics, tabulate_runs


def greit_method(mesh, el_pos, ex_mat, measured_data, step):
    fwd = Forward(mesh, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh["perm"])
    ref_data = f0.v 

    eit = greit.GREIT(mesh, el_pos, ex_mat=ex_mat, step=step, jac_normalized=True, parser="std")
    eit.setup(p=0.50, n=(200, 100), lamb=0.001)
    
    tstart = time.time()

    ds = eit.solve(measured_data, ref_data, normalize=True)
    x, y, ds = eit.mask_value(ds, mask_value=np.NAN)
   
    # suppose xg, yg are the grids along x- and y- axis, eit_im is the greit image
    ds = ds.reshape(x.shape)
    
    pred_perm = np.real(ds)
    run_time = time.time() - tstart 
    
    return pred_perm, run_time


def fix_color_scale(image, range=[1, 0]):
    fixed_image = np.interp(image, [np.min(image), np.max(image)], range)
    fixed_image_smoothed = fixed_image.copy()
    fixed_image_smoothed = np.abs((fixed_image_smoothed -1))
    return fixed_image_smoothed


def run(data_loader, mesh_obj, el_pos, ex_mat, plot, output_dir, mesh_params, device):
    predictions = []
    ground_truth = torch.tensor([])

    avg_run_time = 0.0
   
    for i, (x, y) in enumerate(tqdm(data_loader)):
       
        batch_size = x["perm"].shape[0]
       
        for j in range(batch_size):
            true_perm = x["perm_xy"][j]  
            print(true_perm.shape)
            
            pred_perm, run_time = greit_method(mesh_obj, el_pos, ex_mat, y["v_b"][j], step=1)        
            pred_perm[np.isnan(pred_perm)] = 0 

            pred_perm_xy = pred_perm
            pred_perm_fixed = fix_color_scale(pred_perm_xy, range=[0, 1])

            if plot:
                draw_grid(pred_perm_xy, f"GREIT Prediction", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", save_path=os.path.join(output_dir, f"pred_{i}_{j}.png"))
                draw_grid(pred_perm_fixed, f"GREIT Prediction", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", colorbar=True, save_path=os.path.join(output_dir, f"pred_fix_{i}_{j}.png"))
                draw_grid(true_perm[0], f"Ground Truth", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", colorbar=True, save_path=os.path.join(output_dir, f"truth_{i}_{j}.png"))

            ground_truth = torch.cat((ground_truth, true_perm))
            predictions.append(pred_perm_fixed)
            
            avg_run_time += run_time 

    
    avg_run_time = avg_run_time / len(data_loader)

    predictions = np.array(predictions)
    ground_truth = torch.tensor(ground_truth)
    
    predictions = predictions.reshape(predictions.shape[0], 1, predictions.shape[1], predictions.shape[2])
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])

    metrics = Metrics(device)
    metrics = metrics.forward(torch.tensor(predictions, device=device), torch.tensor(ground_truth, device=device))

    return metrics, predictions, ground_truth, avg_run_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to yaml config file.", default="config/experiments/baselines/greit_exp_bead.yaml")  
    parser.add_argument('--num_runs', type=int, help="Number of runs to report std", default=1) 
    parser.add_argument('--output', type=str, help="Path to output directory.", default="synthetic/baseline") 

    args = parser.parse_args()
    
    num_runs = args.num_runs
    output_dir = args.output 

    config = combine_cfgs(args.config)

    seed = config.SEED 
    batch_size = config.DATASET.BATCH_SIZE
    dataset_path = config.DATASET.PATH
    num_electrodes = config.DATASET.NUM_ELECTRODES
    pos_value = config.DATASET.POS_VALUE
    neg_value = config.DATASET.NEG_VALUE
    normalize = config.DATASET.NORMALIZE 
    shuffle = config.DATASET.SHUFFLE
    standardize = config.DATASET.STANDARDIZE 
    smooth = config.DATASET.SMOOTH
    noise = config.DATASET.NOISE 
    noise_stdv = config.DATASET.NOISE_STDV
    train_min = config.DATASET.TRAIN_MIN
    train_max = config.DATASET.TRAIN_MAX
    train_split, val_split, test_split = config.DATASET.TRAIN_VAL_TEST_SPLIT    

    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Don't do any preprecossing on the data 
    dataset = Dataset(dataset_path, shuffle=shuffle, normalize=normalize, standardize=standardize, pos_value=pos_value, neg_value=neg_value, smooth=smooth, train_max=train_max, train_min=train_min)
    
    split_dataset = not (train_split == 0) 
    
    if split_dataset:
        train_length = int(len(dataset)*train_split)
        val_length = int((len(dataset)*val_split))
        test_length = int((len(dataset) - train_length - val_length))
        _, _, test_dataset = torch.utils.data.random_split(dataset, [train_length, val_length, test_length], generator=torch.Generator().manual_seed(seed))
    else: 
        test_dataset = dataset 

    data_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    ## Construct FEM Mesh ## 
    mesh_params = MeshParams()
    mesh_params.number_electrodes = num_electrodes

    mesh_obj, el_pos = create_mesh(mesh_params=mesh_params)
    
    draw_perm(mesh_obj["node"], mesh_obj["element"], mesh_params,  mesh_obj["perm"], el_pos,  mesh_params.number_electrodes, os.path.join(output_dir, "mesh.png"))

    # Use the same excitation pattern used in the experimental data
    ex_mat = create_ex_mat(mesh_params, max_row_offset=5)
    
    run_metrics = []
    run_avg_time = []

    for k in range(num_runs):
        metrics, predictions, ground_truth, run_time = run(data_loader, mesh_obj, el_pos, ex_mat, plot=(k==0), output_dir=output_dir, mesh_params=mesh_params, device=device)

        run_metrics.append(metrics)
        run_avg_time.append(run_time)
        
        save_path = os.path.join(output_dir, f"predictions_run_{k}.npy")
        with open(save_path, 'wb') as f:
            np.save(f, predictions)
        
        save_path = os.path.join(output_dir, f"ground_truth_run_{k}.npy")
        with open(save_path, 'wb') as f:
            np.save(f, ground_truth)


    save_path = os.path.join(output_dir, "metrics.json")
    stats, table = tabulate_runs(run_metrics, run_avg_time, save_path)
    
    print(table.draw())


if __name__ == "__main__":
    main()