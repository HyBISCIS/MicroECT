# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import os 
import time 
import argparse 

import torch 
import numpy as np

from tqdm import tqdm
import pyeit.eit.jac as jac
from torch.utils.data import DataLoader

from config import combine_cfgs
from data.dataset import Dataset
from data.mesh_params import MeshParams
from data.generate_data import create_mesh, create_ex_mat
from data.plot import interpolate_perm, draw_bp, draw_grid, draw_perm, draw_line 

from metrics.metrics import Metrics, tabulate_runs


def gauss_newton(mesh, el_pos, ex_mat, measured_data, step, perm=1.0, max_iter=1):
    """
        Returns re-cosntructed conductivites using gauss newton (iterated tikhonov regularization)
    """
    
    print("Running Tikhonov Regularization with Gauss Newton Solver...")
    
    eit = jac.JAC(mesh, el_pos, ex_mat, perm=perm, parser="std")
    eit.setup(p=0.5, lamb=10.0, method="lm")
   
    tstart = time.time()
    
    # lamb = lamb * lamb_decay
    ds = eit.gn(measured_data, lamb_decay=0.1, lamb_min=1e-5, maxiter=max_iter, verbose=True)
    run_time = time.time() - tstart 

    return ds, run_time 


def fix_color_scale(image, range=[1, 0]):
    fixed_image = np.interp(image, [np.min(image), np.max(image)], range)
    fixed_image_smoothed = fixed_image.copy()
    return fixed_image_smoothed


def run(data_loader, mesh_obj, el_pos, ex_mat, max_iter, max_len, plot, output_dir, mesh_params, device):
    predictions = []
    ground_truth = torch.tensor([])

    avg_run_time = 0.0
    
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    
    num_runs = 0
    for i, (x, y) in enumerate(tqdm(data_loader)):
        batch_size = x["perm"].shape[0]

        if i != 6: 
            print(f"Skip : {i}")
            continue 

        for j in range(batch_size):

            if j != 21:
                continue 

            true_perm_xy = x["perm_xy"][j]

            # vb_processed = preprocess_capacitence(y["v_b"][j], ex_mat)
            
            # draw_line(np.arange(0, len(vb_processed)), y["v_b"][j], "", "", "", os.path.join(output_dir, f"vb_processed_{j}_{i}.png"))
            
            pred_perm, run_time = gauss_newton(mesh_obj, el_pos, ex_mat, y["v_b"][j], step=1, perm=1.0, max_iter=max_iter)        
            
            pred_perm_xy, _ = interpolate_perm(pts, tri, pred_perm, mesh_params) 
            pred_perm_xy = fix_color_scale(pred_perm_xy)
            
            if plot:
                draw_grid(pred_perm_xy, f"ECT Prediction",  xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", colorbar=False, font_size=18, save_path=os.path.join(output_dir, "pred", f"pred_{i}_{j}.png"))
                draw_grid(pred_perm_xy, f"ECT Prediction",  xlabel="Row (y)", ylabel="Depth (z)", colorbar=False, scale_bar=True, ticks=False, font_size=24, save_path=os.path.join(output_dir, "pred", f"pred_scale_bar_{i}_{j}.png"))

                draw_grid(true_perm_xy[0], f"Ground Truth", xlabel="Row (200\u03bcm)", ylabel="Depth (100\u03bcm)", colorbar=False, font_size=18, save_path=os.path.join(output_dir, "truth", f"truth_{i}_{j}.png"))

            ground_truth = torch.cat((ground_truth, true_perm_xy))
            predictions.append(pred_perm_xy)
            
            avg_run_time += run_time 

            num_runs += 1
        #     print(num_runs, max_len)
        #     if num_runs >= max_len: # limit the #of examples because run-time is huge
        #         break 
        
        # if num_runs >= max_len: # limit the #of examples because run-time is huge
        #     break  
    
    avg_run_time = avg_run_time / len(data_loader)

    predictions = np.array(predictions, dtype=np.float32)
    ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

    predictions = predictions.reshape(predictions.shape[0], 1, predictions.shape[1], predictions.shape[2])
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])

    metrics = Metrics(device=device)
    metrics = metrics.forward(torch.tensor(predictions, device=device), torch.tensor(ground_truth, device=device))

    return metrics, predictions, ground_truth, avg_run_time


def preprocess_capacitence(vb, ex_mat):
    # do the same scalings as the biofilm scaling in the original scripts 
    num_measurements = len(vb)
    my_slice_y = {}
    vb_processed = []
    index = 0
    
    for pattern in ex_mat:
        dist = pattern[1] - pattern[0]
        if str(dist) in my_slice_y.keys():
            my_slice_y[str(dist)].append(vb[index])
        else:
            my_slice_y[str(dist)] = [vb[index]]
        index += 1
    
    for key in my_slice_y.keys():
        slice = [x*10 for x in my_slice_y[key]]
        slice = torch.tensor(slice) - slice[-1]
        vb_processed.extend(slice)

    return vb_processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to yaml config file.", default="config/experiments/baselines/jac_exp_bead.yaml")  
    parser.add_argument('--max_len', type=int, help="Maximum Length of the dataset", default=-1) 
    parser.add_argument('--num_runs', type=int, help="Number of runs to report std", default=1) 
    parser.add_argument('--max_iter', type=int, help="Maximum number of iterations", default=1) 
    parser.add_argument('--output', type=str, help="Path to output directory.", default="synthetic/baseline") 

    args = parser.parse_args()
    
    num_runs = args.num_runs
    max_iter = args.max_iter
    max_len = args.max_len 
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
    os.makedirs(os.path.join(output_dir, "pred"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "truth"), exist_ok=True)

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
    mesh_params.mesh_width = 200e-6
    mesh_params.mesh_height = 100e-6
    mesh_params.mesh_size = 3e-6
    mesh_params.number_electrodes = num_electrodes
    mesh_params.electrode_spacing = 10e-6
    mesh_params.offset = 0

    mesh_obj, el_pos = create_mesh(mesh_params=mesh_params)
    
    draw_perm(mesh_obj["node"], mesh_obj["element"], mesh_params,  mesh_obj["perm"], el_pos,  mesh_params.number_electrodes, os.path.join(output_dir, "mesh.png"))

    # Use the same excitation pattern used in the experimental data
    ex_mat = create_ex_mat(mesh_params, max_row_offset=5)
    
    run_metrics = []
    run_avg_time = []

    for k in range(num_runs):
        metrics, predictions, ground_truth, run_time = run(data_loader, mesh_obj, el_pos, ex_mat, max_iter, max_len, plot=(k==0), output_dir=output_dir, mesh_params=mesh_params, device=device)
        
        run_metrics.append(metrics)
        run_avg_time.append(run_time)
            
        save_path = os.path.join(output_dir, f"predictions_run_{k}.npy")
        with open(save_path, 'wb') as f:
            np.save(f, predictions)
        
        save_path = os.path.join(output_dir, f"ground_truth_run_{k}.npy")
        with open(save_path, 'wb') as f:
            np.save(f, ground_truth)


    save_path = os.path.join(output_dir, "stats.json")
    stats, table = tabulate_runs(run_metrics, run_avg_time)

    print(table.draw())


if __name__ == "__main__":
    main()