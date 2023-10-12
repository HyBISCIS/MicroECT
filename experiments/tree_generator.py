# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os


class TreeGenerator:

    def __init__(self, root_dir):
        self.root_dir = root_dir 

        # root directories
        self.pred_path = os.path.join(self.root_dir, "pred")
        self.true_path = os.path.join(self.root_dir, "truth")
        self.ckp_path = os.path.join(self.root_dir, "ckp")

        # saved model
        self.best_model_path  = os.path.join(self.root_dir, 'best_model.pth')

    def generate(self):
        for dir in [self.root_dir, self.pred_path, self.true_path, self.ckp_path]:
            os.makedirs(dir, exist_ok=True)
    
        with open(self.best_model_path, 'w') as fp:
            pass        
