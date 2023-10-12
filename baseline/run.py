# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

"""
    Run all baaselines 
"""

import os 
import time 
import argparse 

import torch
import numpy as np

from pyunlocbox import functions
from pyunlocbox import solvers
from torch.utils.data import DataLoader

# pyEIT 2D algorithms modules
import pyeit.eit.jac as jac
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit
from pyeit.eit.fem import Forward
from pyeit.mesh import quality

from data.dataset import Dataset
from data.mesh_params import MeshParams
from data.generate_data import create_mesh, create_ex_mat
from data.plot import draw_grid, interpolate_perm, draw_perm, draw_greit, draw_bp
from data.utils import tabulate_runs 
from metrics.metrics import Metrics, tabulate_runs

from bp import run as run_bp 
from greit import run as run_greit
from jac import run as run_jac 
from tv import run as run_tv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Path to dataset directory.", default="logs/04112022-set0-perm-0.25") 
    parser.add_argument('--num_runs', type=int, help="Number of runs to report std", default=1) 
    parser.add_argument('--baselines', nargs='+', help="List of baselines to run", default=["BP"]) 
    parser.add_argument('--gn_iter', type=int, help="Maximum Number of iterations for Gauss Newton", default=1) 
    parser.add_argument('--tv_iter', type=int, help="Maximum Number of iterations for Total Variation", default=100) 
    parser.add_argument('--batch_size', type=int, help="Batch Size", default=1) 
    parser.add_argument('--output', type=str, help="Path to output directory.", default="synthetic/baseline") 

    args = parser.parse_args()

    num_runs = args.num_runs
    gn_iter = args.gn_iter
    tv_iter = args.tv_iter
    baselines = args.baselines
    batch_size = args.batch_size
    dataset_path = args.dataset 
    output_dir = args.output 

    os.makedirs(output_dir, exist_ok=True)
   
    truth_output_dir = os.path.join(output_dir, "truth")
    tv_output_dir = os.path.join(output_dir, "tv")
    gn_output_dir = os.path.join(output_dir, "tk")
    greit_output_dir = os.path.join(output_dir, "greit")
    bp_output_dir = os.path.join(output_dir, "bp")


    os.makedirs(gn_output_dir, exist_ok=True)
    os.makedirs(truth_output_dir, exist_ok=True)
    os.makedirs(tv_output_dir, exist_ok=True)
    os.makedirs(greit_output_dir, exist_ok=True)
    os.makedirs(bp_output_dir, exist_ok=True)

    baselines_dict = {
        "TK": run_jac,
        "TV": run_tv,
        "GREIT": run_greit,
        "BP": run_bp
    }

    # Don't do any preprecossing on the data 
    dataset = Dataset(dataset_path, shuffle=False, normalize=False, noise=False, standardize=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    ## Construct FEM Mesh ## 
    mesh_obj, el_pos = create_mesh(mesh_params=MeshParams)
    
    # Use the same excitation pattern used in the experimental data
    ex_mat = create_ex_mat(MeshParams, max_row_offset=5)

    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    
    quality.stats(pts, tri)
    
    gn_run_time = []
    tv_run_time = []
    greit_run_time = []

    tv_runs = []
    gn_runs = []
    greit_runs = []


    for i, (x, _) in enumerate(data_loader):
        for j in range(batch_size):
            true_perm = x["perm"][j]
            perm_xy, _ = interpolate_perm(pts, tri, true_perm, MeshParams) 
            draw_grid(perm_xy, f"Ground Truth", os.path.join(truth_output_dir, f"truth_{i}_{j}.png"))

    for i in range(num_runs):
        
        plot = (i==0)

        if "TV" in baselines: 
            tv_metrics, tv_time = run_tv(data_loader, batch_size, mesh_obj, el_pos, ex_mat, tv_iter, tv_output_dir, plot) 
            tv_run_time.append(tv_time)
            tv_runs.append(tv_metrics)
      
            print(tv_metrics)

        if "TK" in baselines: 
            gn_metrics, gn_time = run_jac(data_loader, batch_size, mesh_obj, el_pos, ex_mat, gn_iter, gn_output_dir, plot) 
            gn_run_time.append(gn_time)
            gn_runs.append(gn_metrics)

            print(gn_metrics)

        if "GREIT" in baselines:
            greit_metrics, greit_time = run_greit(data_loader, batch_size, mesh_obj, el_pos, ex_mat, None, greit_output_dir, plot)
            
            greit_run_time.append(greit_time)
            greit_runs.append(greit_metrics)

        if "BP" in baselines: 
            bp_metrics, bp_time = run_bp(data_loader, batch_size,  mesh_obj, el_pos, ex_mat, None, bp_output_dir, plot)

   
    tv_mean_time, tv_std_time = np.mean(tv_time), np.std(tv_time)
    gn_mean_time, gn_std_time = np.mean(gn_time), np.std(gn_time)
    
    save_path = os.path.join(output_dir, "tv.json")
    tv_stats, table = tabulate_runs(tv_runs, tv_run_time, save_path)
    save_path = os.path.join(output_dir, "gn.json")
    gn_stats, table = tabulate_runs(gn_runs, gn_run_time, save_path)

    print(tv_stats)
    print(gn_stats)

    print(f"Gauss Newton Mean Time {gn_mean_time}, +/- {gn_std_time}")
    print(f"Gauss Newton Mean Time {tv_mean_time}, +/- {tv_std_time}")


if __name__ == "__main__":
    main()