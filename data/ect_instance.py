
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os
import numpy as np

import pyeit.eit.jac as jac

try: 
    from mesh_params import electrode_pos_to_idx
    from plot import (draw_electric_field, draw_grid, draw_perm, draw_voltage, draw_line)
except Exception as e:
    from .mesh_params import electrode_pos_to_idx
    from .plot import (draw_electric_field, draw_grid, draw_perm, draw_voltage, draw_line)


class ECTInstance:
    
    def __init__(self, index, num_beads, perm, perm_xy, dperm, ext_mat):
        self.index = index
        self.num_beads = num_beads
       
        # permittivity 
        self.perm = perm
        self.perm_xy = perm_xy
        self.dperm = dperm

        # excitation matrix
        self.ext_mat = ext_mat 

        # potential
        self.voltage = []
        self.voltage_xy = []
        self.electric_field = []

        self.voltage_paths = []
        self.voltage_xy_paths = []
        self.dvoltage_paths = []
        
        # boundary voltage 
        self.v_b = []

        # excitation electrodes
        self.ext_electrode = [] 
        self.ext_electrode_pos = []

        self.num_excitations = len(self.ext_electrode)

    def write(self, output_tree):
        
        # write permittivity
        perm_file_name =  f"{self.index}__beads_{self.num_beads}.npy"
        self.perm_path = os.path.join(output_tree.perm_dir, perm_file_name)
        self.perm_xy_path = os.path.join(output_tree.perm_xy_dir, perm_file_name)
        self.dperm_path = os.path.join(output_tree.dperm_dir, perm_file_name)

        for path, arr in zip([self.perm_path, self.perm_xy_path, self.dperm_path], [self.perm, self.perm_xy, self.dperm]):
            with open(path, 'wb') as f:
                np.save(f, np.asarray(arr))

        # write boundary voltage measurement
        self.vb_path = os.path.join(output_tree.vb_dir, f"{self.index}__beads_{self.num_beads}.npy")
        with open(self.vb_path, 'wb') as f:
                np.save(f, np.asarray(self.v_b))

        # write potential measurement
        self.voltage_paths = []
        self.voltage_xy_paths = []
        self.dvoltage_paths = []

        for i in range(self.num_excitations):
            voltage_file_name = f"{self.index}__ext_{self.ext_electrode[i][0]}_{self.ext_electrode[i][1]}.npy"
            
            voltage_path = os.path.join(output_tree.u_dir, voltage_file_name)
            voltage_xy_path = os.path.join(output_tree.u_xy_dir, voltage_file_name)
            dvoltage_path = os.path.join(output_tree.du_dir, voltage_file_name)

            for path, arr in zip([voltage_path, voltage_xy_path, dvoltage_path], [self.voltage[i], self.voltage_xy[i], self.electric_field[i]]):
                with open(path, 'wb') as f:
                    np.save(f, np.asarray(arr))

            self.voltage_paths.append(voltage_path)
            self.voltage_xy_paths.append(voltage_xy_path)
            self.dvoltage_paths.append(dvoltage_path)


    def add_solution(self, ext_electrode, ext_electrode_pos, voltage, voltage_xy, electric_field):
        self.ext_electrode.append(ext_electrode)
        self.ext_electrode_pos.append(ext_electrode_pos)
        self.num_excitations = len(self.ext_electrode)
        self.voltage.append(voltage)
        self.voltage_xy.append(voltage_xy)
        self.electric_field.append(electric_field)


    def plot(self, mesh_points, mesh_triangles, mesh_params, el_pos, output_tree, debug):
        perm_file_name =   f"{self.index}__beads_{self.num_beads}.png"
        perm_path = os.path.join(output_tree.perm_dir, perm_file_name)
        perm_xy_path = os.path.join(output_tree.perm_xy_dir, perm_file_name)
        dperm_path = os.path.join(output_tree.dperm_dir, perm_file_name)
        est_perm_path = os.path.join(output_tree.est_perm_dir, perm_file_name)

        # draw permittivity
        draw_grid(self.perm_xy, "interpolated permittivity (rectangular grid)", xlabel="x", ylabel="Depth (z)", save_path=perm_xy_path) 
            
        if len(self.perm) > 0 : 
            draw_perm(mesh_points, mesh_triangles, mesh_params, self.perm, el_pos, mesh_params.number_electrodes, perm_path) 

        if debug and len(self.dperm) > 0:
            dperm_mag = np.sqrt(self.dperm[0]**2 + self.dperm[1]**2)
            # draw_perm(mesh_points, mesh_triangles, mesh_params, self.estimated_perm, el_pos, mesh_params.number_electrodes, est_perm_path) 
            draw_grid(dperm_mag, "permittivity derivative", xlabel="x", ylabel="Depth (z)", save_path=dperm_path)

        # draw boundary voltage
        vb_file_name = os.path.join(output_tree.vb_dir, f"{self.index}__beads_{self.num_beads}.png")
        draw_line(np.arange(0, len(self.v_b)), self.v_b, "boundary measurement", "Measurement" , "Capacitence", vb_file_name)

        for i in range(len(self.ext_electrode)):
            voltage_file_name = f"{self.index}__ext_{self.ext_electrode[i][0]}_{self.ext_electrode[i][1]}.png"
            dvoltage_xy_file_name = f"{self.index}__ext_{self.ext_electrode[i][0]}_{self.ext_electrode[i][1]}_xy.png"

            voltage_path = os.path.join(output_tree.u_dir, voltage_file_name)
            voltage_xy_path = os.path.join(output_tree.u_xy_dir, voltage_file_name)
            dvoltage_path = os.path.join(output_tree.du_dir, voltage_file_name)
            dvoltage_xy_path = os.path.join(output_tree.du_dir, dvoltage_xy_file_name)

            draw_grid(self.voltage_xy[i],  'Interpolated Potential Distribution' , voltage_xy_path)

            if debug:             
                # potential distribution
                draw_voltage(mesh_points, mesh_triangles, mesh_params, self.perm, self.ext_electrode[i], self.voltage[i], el_pos, voltage_path)
                
                # electric field
                e_mag = np.sqrt(self.electric_field[i][0]**2 + self.electric_field[i][1]**2)
                draw_electric_field(mesh_points, mesh_triangles, mesh_params, self.perm, self.electric_field[i], el_pos, dvoltage_xy_path)
                draw_grid(e_mag, 'Electric Field Distribution', dvoltage_path)


    def check_solution(self, mesh_params, output_tree):
        # pde residual 
        dperm_x, dperm_y = self.dperm

        for i in range(len(self.ext_electrode)):
            du_x, du_y = self.electric_field[i]
            d2u_xx, _ = np.gradient(du_x)
            _, d2u_yy = np.gradient(du_y)

            pde = dperm_x * du_x + dperm_y * du_y + self.perm_xy * (d2u_xx + d2u_yy)

            # plot pde error
            draw_grid(pde, 'PDE Residual', os.path.join(output_tree.pde_dir, f"{self.index}__{i}.png"))            

            # Natural boundary condition (Neumann B.C)
            x1_idx, y1_idx = electrode_pos_to_idx(self.ext_electrode_pos[i][0, :])
            x2_idx, y2_idx = electrode_pos_to_idx(self.ext_electrode_pos[i][1, :])

            bottom_normal, top_normal = ([0, -1], [0, 1])
            right_normal, left_normal = ([-1, 0], [1, 0])

            # inner product of the gradient vector with the normal vector
            bc_1 = self.perm_xy[0][x1_idx] * np.inner((du_x[0][x1_idx], du_y[0][x1_idx]), bottom_normal)
            bc_2 = self.perm_xy[0][x2_idx] * np.inner((du_x[0][x2_idx], du_y[0][x2_idx]), bottom_normal)

            # print("Boundary 1: ", y1_idx, bc_1)
            # print("Boundary 2: ", y2_idx, bc_2)
            
            neumann_bc = np.zeros(self.perm_xy.shape)
            # top
            neumann_bc[-1, :] = self.perm_xy[-1, :] * np.sum(np.array((du_x[-1, :], du_y[-1, :])).T * np.tile(top_normal, (self.perm_xy.shape[1], 1)), axis=1)
            # bottom 
            neumann_bc[0, :] = self.perm_xy[0, :] * np.sum(np.array((du_x[0, :], du_y[0, :])).T * np.tile(bottom_normal, (self.perm_xy.shape[1], 1)), axis=1)
            # left
            neumann_bc[:, 0] = self.perm_xy[:, 0] * np.sum(np.array((du_x[:, 0], du_y[:, 0])).T * np.tile(left_normal, (self.perm_xy.shape[0], 1)), axis=1)
            # right
            neumann_bc[:, -1] = self.perm_xy[0, -1] * np.sum(np.array((du_x[:, -1], du_y[:, -1])).T * np.tile(right_normal, (self.perm_xy.shape[0], 1)), axis=1)

            draw_grid(neumann_bc, 'Neumann BC', os.path.join(output_tree.bc_dir, f"{self.index}__{i}.png"))            

            # Integration Condition 
            sum_u = np.sum(self.voltage_xy[i][-1, :]) + np.sum(self.voltage_xy[i][0, :]) + np.sum(self.voltage_xy[i][:, 0]) + np.sum(self.voltage_xy[i][:, -1])
            # print(sum_u)

            sum_u = np.sum(np.sum(self.voltage_xy[i]))
            # print(sum_u)

    def estimate_permittivity(self, mesh_obj, el_pos, ex_mat):
        eit = jac.JAC(mesh_obj, el_pos, ex_mat, perm=1.0, parser="std")
        eit.setup(p=0.25, lamb=10.0, method="lm")
        self.estimated_perm = eit.gn(self.v_b, lamb_decay=0.1, lamb_min=1e-5, maxiter=10, verbose=True)
       

    def dict(self, root_dir, is_exp=False):
        self.voltage_dict = {}

        for i in range(self.num_excitations):
            u = {
                "ext_elec": self.ext_electrode[i].tolist(),
                "ext_elec_pos": self.ext_electrode_pos[i].tolist(),
                "u":  os.path.relpath(self.voltage_paths[i], root_dir),
                "u_xy": os.path.relpath(self.voltage_xy_paths[i], root_dir),
                "du": os.path.relpath(self.dvoltage_paths[i], root_dir),
            }
            self.voltage_dict.update({f"{i}": u})

        self.data_dict = {
                "index" : self.index,
                "is_exp": is_exp,
                "num_beads": self.num_beads,
                "perm": os.path.relpath(self.perm_path, root_dir),
                "perm_xy": os.path.relpath(self.perm_xy_path, root_dir), 
                "dperm" :  os.path.relpath(self.dperm_path, root_dir), 
                "num_ext": self.num_excitations,
                "v_b": os.path.relpath(self.vb_path, root_dir),           # boundary measurement
                "u": self.voltage_dict      # potential distribution
        }

        return self.data_dict