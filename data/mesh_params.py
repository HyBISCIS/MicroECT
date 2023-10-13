# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  


import numpy as np 
from dataclasses import dataclass

@dataclass
class MeshParams:
    mesh_width: float = 200e-6
    mesh_height: float = 100e-6
    mesh_size: float = 3e-6
    electrode_spacing: float = 10e-6
    electrod_offset: float = 50e-6
    number_electrodes: int = 15 
    offset: float = 0
    # -10e-6

    # non-microns 
    absolute_height: int = (mesh_height + np.abs(offset))*1e6
    absolute_width : int = (mesh_width)*1e6


def get_2d_grid(mesh_params): 
    rgrid_size=1e-6
    x_rgrid, y_rgrid = np.meshgrid(np.arange(-(mesh_params.mesh_width/2), (mesh_params.mesh_width/2)-1e-6, rgrid_size),
                              np.arange(mesh_params.offset, mesh_params.mesh_height-1e-6, rgrid_size))

    return (x_rgrid, y_rgrid)


def generate_normal_vectors(repeat=True):
    bottom_normal, top_normal = ([0, -1], [0, 1])
    right_normal, left_normal = ([-1, 0], [1, 0])
    
    if repeat: 
        bottom_normal = np.tile(bottom_normal, (MeshParams.absolute_width, 1))
        top_normal = np.tile(top_normal, (MeshParams.absolute_width, 1))
        right_normal = np.tile(right_normal, (MeshParams.absolute_height, 1))
        left_normal = np.tile(left_normal, (MeshParams.absolute_height, 1))

    return {"bottom": bottom_normal, "top": top_normal, "left": left_normal, "right": right_normal}


def electrode_pos_to_idx(electrode_pos):
    x, y = electrode_pos[0], electrode_pos[1]
    x_idx = (x + (MeshParams.mesh_width / 2)) * 1e6
    y_idx = (MeshParams.mesh_height - np.abs(MeshParams.offset) - y - 1e-6) / 1e-6
    return int(x_idx), int(y_idx)
