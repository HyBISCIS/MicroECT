 
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os
import json 
from json.decoder import JSONDecodeError

class TreeGenerator:

    def __init__(self, root_dir):
        self.root_dir = root_dir 

        # root directories
        self.x_root = os.path.join(self.root_dir, "x")
        self.y_root = os.path.join(self.root_dir, "y")
        self.debug_root = os.path.join(self.root_dir, "debug")

        # permittivity output directories
        self.perm_dir = os.path.join(self.x_root, "perm")
        self.perm_xy_dir = os.path.join(self.x_root, "perm_xy")
        self.dperm_dir = os.path.join(self.x_root, "dperm")

        self.x_dirs = [self.perm_dir, self.perm_xy_dir, self.dperm_dir]

        # potential output directories
        self.u_dir = os.path.join(self.y_root, "u")
        self.u_xy_dir = os.path.join(self.y_root, "u_xy")
        self.du_dir = os.path.join(self.y_root, "du")

        # boundary voltage output directories 
        self.vb_dir = os.path.join(self.y_root, "v_b")

        self.y_dirs = [self.u_dir, self.u_xy_dir, self.du_dir, self.vb_dir]

        # solution debugging output directories
        self.pde_dir = os.path.join(self.debug_root, "pde")
        self.bc_dir = os.path.join(self.debug_root, "bc")
        self.est_perm_dir = os.path.join(self.debug_root, "est_perm")
        self.exp_dir = os.path.join(self.debug_root, "exp_logs")

        self.debug_dirs = [self.pde_dir, self.bc_dir, self.est_perm_dir, self.exp_dir]

        # dataset.json file
        self.dataset_json = os.path.join(self.root_dir, "dataset.json")
    
    def generate(self):
        for dir in [self.root_dir, self.x_root, self.y_root, self.debug_root]:
            os.makedirs(dir, exist_ok=True)

        for dir in self.x_dirs + self.y_dirs + self.debug_dirs:
            os.makedirs(dir, exist_ok=True)

        with open(self.dataset_json, 'w') as fp:
            pass        

    def write_json(self, dict):
        # read the json file first
        # data = {}
        # try:
        #     json_file = open(self.dataset_json, "r") 
        #     data = json.load(json_file) 
        #     json_file.close()
        # except JSONDecodeError:
        #     pass

        # # update data dict 
        # data.update(dict)

        json_file = open(self.dataset_json, "a")
        json_file.write(json.dumps(dict, indent=4))
        # json_file.write(json.dumps(data, indent=4))
        json_file.close()


