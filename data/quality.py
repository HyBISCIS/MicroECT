
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  


import os 
import math 
import time

import argparse
import numpy as np 
from mesh_params import MeshParams 


import pyeit.mesh as mesh
from pyeit.eit.interp2d import tri_area, sim2pts
from pyeit.mesh import quality
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac

from plot import draw_line 


def mesh_quality(pts, tri):
    """ Compute Triangle Mesh Quality """
    avg_quality = 0 

    def dist(p1, p2):
        return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    for triangle in tri: 
        v1 = pts[triangle[0]]
        v2 = pts[triangle[1]]
        v3 = pts[triangle[2]]

        a = dist(v1, v2)
        b = dist(v2, v3)
        c = dist(v1, v3)
        
        sum_sides_sq = (a**2 + b**2 + c**2)

        # calculate the semi-perimeter
        s = (a + b + c) / 2

        # calculate the area
        area = (s*(s-a)*(s-b)*(s-c)) ** 0.5

        avg_quality += area / sum_sides_sq

    avg_quality = avg_quality / len(tri)
    total_quality = avg_quality * (12 / 3**0.5)
    
    return total_quality 


def convergence_error(curr_solution, prev_solution):
    """Compute Convergence Error for the FEM Solution"""
    e_c = (np.abs(curr_solution) - np.abs(prev_solution) ) / np.abs(prev_solution)
    e_c = np.max(e_c)
    return e_c 


def build_mesh(mesh_params): 
    def myrectangle(pts):
        return mesh.shape.rectangle(pts,p1=[-mesh_params.mesh_width/2,0],p2=[mesh_params.mesh_width/2, mesh_params.mesh_height])
   
    p_fix = np.array([[x,0] for x in np.arange(-(mesh_params.number_electrodes//2*mesh_params.electrode_spacing), (mesh_params.number_electrodes//2+1)*mesh_params.electrode_spacing,mesh_params.electrode_spacing)])  # electrodes
    p_fix = np.append(p_fix, np.array([[x, mesh_params.mesh_width] for x in np.arange(-mesh_params.mesh_width/2, mesh_params.mesh_width/2, mesh_params.mesh_size)]), axis=0)   # dirichlet nodes (const voltage)

    mesh_obj, el_pos = mesh.create(len(p_fix), 
                                fd=myrectangle, 
                                p_fix=p_fix, 
                                h0=mesh_params.mesh_size,
                                bbox = np.array([[-mesh_params.mesh_width/2, 0], [mesh_params.mesh_width/2, mesh_params.mesh_height]]),
                                )
    
    ex_mat = []
    for i in range(MeshParams.number_electrodes-1):
        ext_pattern = [[x, x+i+1] for x in np.arange(0, MeshParams.number_electrodes-i-1, 1)]
        ex_mat.extend(ext_pattern)
    
    ex_mat = np.array(ex_mat)

    return mesh_obj, el_pos, ex_mat


def refine_mesh_size(mesh_sizes):
    quality_metric = []
    conv_error = []
    run_time = []
    mesh_params = MeshParams()    

    prev_solution = None 

    for size in mesh_sizes: 
        mesh_params.mesh_size = size
        start_time = time.time() 
        mesh_obj, el_pos, ex_mat = build_mesh(mesh_params)
        end_time = time.time()
        total_time = end_time - start_time 

        pts = mesh_obj["node"]
        tri = mesh_obj["element"]
        x, y = pts[:, 0], pts[:, 1]
        quality.stats(pts, tri)

        tri = np.array(tri)
        avg_quality = mesh_quality(pts, tri)
        
        anomaly = [{"x": 0e-6, "y": 10e-6, "d": 20e-6, "perm": 0.25}]
        mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
        
        fwd = Forward(mesh_obj, el_pos)
        f1 = fwd.solve_eit(ex_mat, perm=mesh_new["perm"], parser="std")

        curr_solution = f1.v[:, np.newaxis]
        
        if prev_solution is not None: 
            error = convergence_error(curr_solution, prev_solution)
            conv_error.append(error)
        
        prev_solution = curr_solution

        run_time.append(total_time)
        quality_metric.append(avg_quality)

    return quality_metric, conv_error, run_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="Path output directorty. ", required=True)

    args = parser.parse_args()

    output_dir = args.output_dir 

    # 0.5e-6, 0.1e-6
    element_size = [10e-6, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6] 
    quality, conv_error, run_time = refine_mesh_size(element_size)

    draw_line(x=element_size, y=quality, title="Mesh Quality", xlabel="Element Size (microns)", ylabel="Quality Metric", save_path=os.path.join(output_dir, "quality.png"))
    draw_line(x=element_size, y=run_time, title="Run Time v.s Element Size", xlabel="Element Size (microns)", ylabel="Run time (s)", save_path=os.path.join(output_dir, "run_time.png"))
    draw_line(x=element_size[1:], y=conv_error, title="Convergence Error v.s Element Size", xlabel="Element Size (microns)", ylabel="Convergence Error", save_path=os.path.join(output_dir, "conv_error.png"))



if __name__ == "__main__":
    main()