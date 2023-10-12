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
import pyeit.eit.bp as bp
from pyeit.eit.fem import Forward
from torch.utils.data import DataLoader

from data.dataset import Dataset
from data.mesh_params import MeshParams
from data.generate_data import create_mesh, create_ex_mat
from data.plot import interpolate_perm, draw_bp, draw_grid 
from metrics.metrics import Metrics, tabulate_runs


def back_projection(mesh, el_pos, ex_mat, measured_data, step=1):
    
    fwd = Forward(mesh, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh["perm"])
    ref_data = f0.v 

    eit = bp.BP(mesh, el_pos, ex_mat=ex_mat, step=1, jac_normalized=True, parser="std")
    eit.setup(weight="none")

    tstart = time.time()

    ds = eit.solve(measured_data, ref_data, normalize=True) 
    pred_perm = np.real(ds)

    run_time = time.time() - tstart 

    return pred_perm, run_time


def run(data_loader, mesh_obj, el_pos, ex_mat, plot, output_dir):
    predictions = []
    ground_truth = []

    avg_run_time = 0.0

    metrics = Metrics()

    pts = mesh_obj["node"]
    tri = mesh_obj["element"] 

    for i, (x, y) in enumerate(tqdm(data_loader)):
        batch_size = x["perm"].shape[0]
        for j in range(batch_size):
         
            true_perm = x["perm"][j]

            pred_perm, run_time = back_projection(mesh_obj, el_pos, ex_mat, y["v_b"][j], step=1)        

            true_perm_xy, _ = interpolate_perm(pts, tri, true_perm, MeshParams) 

            if plot:
                draw_bp(pred_perm, pts, tri, true_perm, os.path.join(output_dir, f"pred_perm_{i}_{j}.png"))
                draw_grid(true_perm_xy, f"Ground Truth", os.path.join(output_dir, f"truth_{i}_{j}.png"))

        ground_truth.append(true_perm_xy)
        predictions.append(pred_perm)
        
        avg_run_time += run_time 

        break 
    
    avg_run_time = avg_run_time / len(data_loader)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # TODO: interpolate back projection predicitons 
    # predictions = predictions.reshape(predictions.shape[0], 1, predictions.shape[1], predictions.shape[2])
    # ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])

    # metrics = metrics.forward(torch.tensor(predictions), torch.tensor(ground_truth))
    metrics = None 
    return metrics, avg_run_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Path to dataset directory.", default="logs/04112022-set0-perm-0.25")  
    parser.add_argument('--num_runs', type=int, help="Number of runs to report std", default=1) 
    parser.add_argument('--batch_size', type=int, help="Batch Size", default=1) 
    parser.add_argument('--output', type=str, help="Path to output directory.", default="synthetic/baseline") 

    args = parser.parse_args()
    
    num_runs = args.num_runs
    batch_size = args.batch_size
    dataset_path = args.dataset 
    output_dir = args.output 

    os.makedirs(output_dir, exist_ok=True)

    # Don't do any preprecossing on the data 
    dataset = Dataset(dataset_path, shuffle=False, normalize=False, noise=False, standardize=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    ## Construct FEM Mesh ## 
    mesh_obj, el_pos = create_mesh(mesh_params=MeshParams)
    
    # Use the same excitation pattern used in the experimental data
    ex_mat = create_ex_mat(MeshParams, max_row_offset=5)


    run_metrics = []
    run_avg_time = []

    for k in range(num_runs):
        metrics, run_time = run(data_loader, mesh_obj, el_pos, ex_mat, plot=(k==0), output_dir=output_dir)
        
        run_metrics.append(metrics)
        run_avg_time.append(run_time)
    
    save_path = os.path.join(output_dir, "tv.json")
    stats, table = tabulate_runs(run_metrics, run_avg_time, save_path)
    print(table.draw())


if __name__ == "__main__":
    main()