
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
    Create Synthetic Dataset from pyEIT 
"""

import random
import argparse
import numpy as np 

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward

try: 
    from tree_generator import TreeGenerator
    from mesh_params import MeshParams
    from ect_instance import ECTInstance
    from random_perm import generate_noise, set_perm
    from plot import interpolate_perm, interpolate_voltage
except Exception as e: 
    from .tree_generator import TreeGenerator
    from .mesh_params import MeshParams
    from .ect_instance import ECTInstance
    from .random_perm import generate_noise, set_perm
    from .plot import interpolate_perm, interpolate_voltage

from skimage.transform import resize

import matplotlib.pyplot as plt


def create_mesh(mesh_params):

    # Mesh shape
    def myrectangle(pts):
        return mesh.shape.rectangle(pts,p1=[-mesh_params.mesh_width/2, mesh_params.offset],p2=[mesh_params.mesh_width/2, mesh_params.mesh_height])

    # Electrodes
    p_fix = np.array([[x,0] for x in np.arange(-(mesh_params.number_electrodes//2*mesh_params.electrode_spacing),(mesh_params.number_electrodes//2+(mesh_params.number_electrodes % 2))*mesh_params.electrode_spacing, mesh_params.electrode_spacing)])  # electrodes
    p_fix = np.append(p_fix, np.array([[x,mesh_params.mesh_width] for x in np.arange(-mesh_params.mesh_width/2,mesh_params.mesh_width/2,mesh_params.mesh_size)]), axis=0)   # fixed electrdes (const voltage)
    
    # Create mesh
    mesh_obj, el_pos = mesh.create(len(p_fix), 
                               fd=myrectangle, 
                               p_fix=p_fix,      
                               h0=mesh_params.mesh_size,
                               bbox = np.array([[-mesh_params.mesh_width/2, 0], [mesh_params.mesh_width/2, mesh_params.mesh_height]]),
                               )
    
    return mesh_obj, el_pos 


def create_ex_mat(mesh_params, max_row_offset=5):
    # Excitation Pattern 
    ex_mat = []
    for i in range(mesh_params.number_electrodes-1):
        if i < max_row_offset: 
            ext_pattern = [[x, x+i+1] for x in np.arange(0, mesh_params.number_electrodes-i-1, 1)]
            ex_mat.extend(ext_pattern)
        else:
            break
    
    ex_mat = np.array(ex_mat) 

    return ex_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help="Path to output directory.", default="data/trial4")
    parser.add_argument('--dataset_size', type=int, help="Number of data points.", default=1000)
    parser.add_argument('--max_num_beads', type=int, help="Maximum number of beads.", default=2)
    parser.add_argument('--max_row_offset', type=int, help="Maximum row offset.", default=5)
    parser.add_argument('--min_bead_diameter', type=int, help="Minimum bead diameter", default=20e-6)
    parser.add_argument('--random_seed', type=int, help="Random seed.", default=0)
    parser.add_argument('--bead_max_y', type=int, help="Maximum y location.", default=0.8*MeshParams.mesh_height)
    parser.add_argument('--overlap_beads', help="Allow beads to overlap to create complex geometries.", default=False, action="store_true")
    parser.add_argument('--debug', help="Save debug informaion like PDE Residual.", default=False, action="store_true")
    parser.add_argument('--full-eit', help="Generate all EIT data including the solution inside mesh.", default=False, action="store_true")
    parser.add_argument('--est-perm', help="Estimate Permittivity using iterative method.", default=False, action="store_true")
    parser.add_argument('--biofilm', help="Generate biofilm-like permittivity distributions.", default=False, action="store_true")

    args = parser.parse_args()
    output_path = args.output_path 
    dataset_size = args.dataset_size 
    max_num_beads = args.max_num_beads
    max_row_offset = args.max_row_offset
    min_bead_diameter = args.min_bead_diameter
    random_seed = args.random_seed
    max_bead_y = args.bead_max_y
    overlap_beads = args.overlap_beads 
    estimate_perm = args.est_perm 
    full_eit = args.full_eit 
    biofilm = args.biofilm
    debug = args.debug 

    np.random.seed(random_seed)
    random.seed(random_seed)

    output_tree = TreeGenerator(output_path)
    output_tree.generate()
    
    # output_tree.write_json({"dataset_size": dataset_size, "max_num_beads": max_num_beads, "random_seed": 0, "data": {}})

    def myrectangle(pts):
        return mesh.shape.rectangle(pts,p1=[-MeshParams.mesh_width/2, MeshParams.offset],p2=[MeshParams.mesh_width/2, MeshParams.mesh_height])

    # Electrodes
    p_fix = np.array([[x,0] for x in np.arange(-(MeshParams.number_electrodes//2*MeshParams.electrode_spacing),(MeshParams.number_electrodes//2+1)*MeshParams.electrode_spacing, MeshParams.electrode_spacing)])  # electrodes
    p_fix = np.append(p_fix, np.array([[x,MeshParams.mesh_width] for x in np.arange(-MeshParams.mesh_width/2,MeshParams.mesh_width/2,MeshParams.mesh_size)]), axis=0)   # fixed electrdes (const voltage)
    
    # Mesh
    mesh_obj, el_pos = mesh.create(len(p_fix), 
                               fd=myrectangle, 
                               p_fix=p_fix,      
                               h0=MeshParams.mesh_size,
                               bbox = np.array([[-MeshParams.mesh_width/2, 0], [MeshParams.mesh_width/2, MeshParams.mesh_height]]),
                               )


    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    perm = mesh_obj["perm"].copy()
    x, y = pts[:, 0], pts[:, 1]

    # Excitation Pattern 
    ex_mat = []
    for i in range(MeshParams.number_electrodes-1):
        if i < max_row_offset: 
            ext_pattern = [[x, x+i+1] for x in np.arange(0, MeshParams.number_electrodes-i-1, 1)]
            ex_mat.extend(ext_pattern)
        else:
            break
    
    ex_mat = np.array(ex_mat)

    num_beads = 0

    for i in range(0, dataset_size):
        if biofilm:
            random_image, _ = generate_noise(MeshParams.absolute_width, MeshParams.absolute_height, max_bead_y*1e6, i)
            mesh_new = mesh.set_perm(mesh_obj, anomaly=[], background=1.0)
            mesh_new["perm"] = set_perm(mesh_new["perm"], tri, pts, random_image)
            perm = mesh_new["perm"].copy()
        else: 
            num_beads = np.random.randint(1, max_num_beads+1)

            max_bead_diameter = 20e-6
            # (min(MeshParams.mesh_width, MeshParams.mesh_height)-10e-6) / num_beads
        
            anomlies = []
            for j in range(0, num_beads):
                bead_diameter = np.random.uniform(min_bead_diameter, max_bead_diameter)
                bead_x = np.random.uniform(-MeshParams.mesh_width/2+bead_diameter/2+25e-6, MeshParams.mesh_width/2-bead_diameter/2-25e-6)
                bead_y = np.random.uniform(bead_diameter/2+1e-6, max_bead_y-bead_diameter/2) 

                # make sure that beads don't overlap
                if not overlap_beads and j != 0: 
                    overlap = True
                    while overlap: 
                        for anomly in anomlies: 
                            distance = np.sqrt((anomly["x"] - bead_x)**2 + (anomly["y"] - bead_y)**2)
                            overlap = distance < (anomly["d"] + bead_diameter + 10e-6)

                        if overlap: 
                            bead_x = np.random.uniform(-MeshParams.mesh_width/2+bead_diameter/2+5e-6, MeshParams.mesh_width/2-bead_diameter/2-5e-6)
                            bead_y = np.random.uniform(bead_diameter/2+10e-6, max_bead_y-bead_diameter/2) 

                bead = {"x": bead_x, "y": bead_y, "d": bead_diameter/2, "perm": 0.25}
                anomlies.append(bead)

            if MeshParams.offset != 0: 
                anomly = {"bbox": np.array([[-MeshParams.mesh_width/2, MeshParams.offset], [MeshParams.mesh_width, 0e-6]]), "perm": 0.5}
                anomlies.append(anomly)   

            mesh_new = mesh.set_perm(mesh_obj, anomaly=anomlies, background=1.0)

            # permittivity value for each triangle in the mesh grid
            perm = mesh_new["perm"] 

        # interpolate permittivity to 2D grid  
        perm_xy, dperm = interpolate_perm(pts, tri, perm, MeshParams)
       
        instance = ECTInstance(index=i, perm=perm, perm_xy=perm_xy, dperm=dperm, num_beads=num_beads, ext_mat=ex_mat)

        ## FEM Forward Simulations ##
        fwd = Forward(mesh_new, el_pos)
        
        # returns potential on nodes ; shape (n_pts,)
        f = fwd.solve_eit(ex_mat, perm=perm,  parser="std")
        sim_voltage = np.real(f.v)
        
        instance.v_b = sim_voltage 
         
        # generate the solution for each stimulation pattern
        if full_eit: 
            for j, ext_pattern in enumerate(ex_mat):
                f, _ = fwd.solve(ext_pattern.ravel(), perm=perm)
                f = np.real(f)

                u_xy, e_xy = interpolate_voltage(pts, tri, f, MeshParams)

                electrode_pos = np.zeros((2, 2))
                electrode_pos[0, :] = [x[el_pos[ext_pattern[0]]], y[el_pos[ext_pattern[0]]]]
                electrode_pos[1, :] = [x[el_pos[ext_pattern[1]]], y[el_pos[ext_pattern[1]]]]

                instance.add_solution(ext_pattern, electrode_pos, f, u_xy, e_xy)

        if debug: 
            instance.check_solution(MeshParams, output_tree)
     
        if estimate_perm: 
            instance.estimate_permittivity(mesh_obj, el_pos, ex_mat)
            
        instance.write(output_tree)
        instance.plot(pts, tri, MeshParams, el_pos, output_tree, debug)

        # update json file
        output_tree.write_json({f"{i}": instance.dict(output_tree.root_dir)})


if __name__ == "__main__":
    main()