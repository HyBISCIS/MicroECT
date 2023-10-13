# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
    Merge multiple dataset directories into one directory
"""
import os
import re
import json
import shutil
import argparse
from pathlib import Path

import numpy as np 

from ect_instance import ECTInstance
from tree_generator import TreeGenerator 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help="Path to output directory.", default="data/DATASET") 
    parser.add_argument('--dataset_paths', nargs='+', help="Path to the dataset directories.", required=True) 

    args = parser.parse_args()
    output_path = args.output_path 
    dataset_paths = args.dataset_paths 

    output_tree = TreeGenerator(output_path)
    output_tree.generate()

    index = 0

    for dataset in dataset_paths:
        json_path = os.path.join(dataset, "dataset.json")
        f = open(json_path)
        data = json.load(f)

        for _, subdict in data.items():
            perm_path = subdict["perm"]
            perm_xy_path = subdict["perm_xy"]
            dperm_path = subdict["dperm"]
            num_beads = subdict["num_beads"]
            num_ext = subdict["num_ext"]
            vb_path = subdict["v_b"]

            new_file_names = []
            for path in [perm_path, perm_xy_path, dperm_path, vb_path]:
                new_file_names.append(re.sub(r'\d+', rf'{index}', Path(path).name , count=1))

            instance = ECTInstance(index=index, num_beads=num_beads, perm=None, perm_xy=None, dperm=None, ext_mat=None)

            instance.perm_path = os.path.join(output_tree.perm_dir, new_file_names[0])
            instance.perm_xy_path = os.path.join(output_tree.perm_xy_dir, new_file_names[1])
            instance.dperm_path =  os.path.join(output_tree.dperm_dir, new_file_names[2])
            instance.vb_path = os.path.join(output_tree.vb_dir, new_file_names[3])
            instance.num_excitations = num_ext

            # copy permittivity files 
            for src, dst in zip([perm_path, perm_xy_path, dperm_path, vb_path], [instance.perm_path, instance.perm_xy_path, instance.dperm_path, instance.vb_path]):
                src_path = os.path.join(dataset, src)
                shutil.copy(src_path, dst)
                if Path(src_path).with_suffix('.png').exists():
                    shutil.copy(Path(src_path).with_suffix('.png'), Path(dst).with_suffix('.png'))
                # else: 
                    # print(f"[Warning]: {Path(src).with_suffix('.png')} doesn't exist.")

            # copy u files 
            solutions = subdict["u"]
            ext_electrode = []
            ext_electrode_pos = []
            for solution in solutions.items():
                _, solution = solution
                ext_elec = solution["ext_elec"]
                ext_elec_pos = solution["ext_elec_pos"]

                ext_electrode.append(ext_elec)
                ext_electrode_pos.append(ext_elec_pos)

                u =  solution["u"] 
                u_xy = solution["u_xy"]
                du = solution["du"]

                new_file_names = []
                for path in [u, u_xy, du]:
                    new_file_names.append(re.sub(r'\d+', rf'{index}', Path(path).name , count=1))

                instance.voltage_paths.append(os.path.join(output_tree.u_dir, new_file_names[0]))
                instance.voltage_xy_paths.append(os.path.join(output_tree.u_xy_dir, new_file_names[1]))
                instance.dvoltage_paths.append(os.path.join(output_tree.du_dir, new_file_names[2]))

                for src, dst in zip([u, u_xy, du], [instance.voltage_paths[-1], instance.voltage_xy_paths[-1], instance.dvoltage_paths[-1]]):
                    src_path = os.path.join(dataset, src)
                    shutil.copy(src_path, dst)
                    shutil.copy(Path(src_path).with_suffix('.png'), Path(dst).with_suffix('.png'))
            
            instance.ext_electrode = np.array(ext_electrode)
            instance.ext_electrode_pos = np.array(ext_electrode_pos)

            output_tree.write_json({f"{index}": instance.dict(output_path)})
            
            index += 1


if __name__ == "__main__":
    main()