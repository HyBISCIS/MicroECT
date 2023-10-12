# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import sys 


from pathlib import Path
from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()
__C.NAME = "exp01"
__C.SEED = 0


# data augmentation parameters with albumentations library
__C.DATASET = ConfigurationNode()
__C.DATASET.PATH = "data/synthetic/data/dataset-small-beads-final"
__C.DATASET.TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
__C.DATASET.NORMALIZE = True
__C.DATASET.NOISE = True
__C.DATASET.NOISE_STDV =  0.03
__C.DATASET.STANDARDIZE = False
__C.DATASET.SHUFFLE = False
__C.DATASET.SMOOTH = False
__C.DATASET.BATCH_SIZE = 32
__C.DATASET.NUM_MEASUREMENTS = 60
__C.DATASET.NUM_ELECTRODES = 15
__C.DATASET.POS_VALUE = -1.0
__C.DATASET.NEG_VALUE = 1.0
__C.DATASET.TRAIN_MIN = None
__C.DATASET.TRAIN_MAX = None

# model head configs
__C.MODEL = ConfigurationNode()
__C.MODEL.TYPE = 'Vanilla-Decoder'   # Residual-Decoder
__C.MODEL.HEAD_ACTIVATION = 'Tanh'
__C.MODEL.HIDDEN_ACTIVATION = 'ReLU'

# Solver 
__C.SOLVER = ConfigurationNode()
__C.SOLVER.LEARNING_RATE = 0.0001
__C.SOLVER.LR_SCHEDULER = "LambdaLR"
__C.SOLVER.LR_GAMMA = 0.85
__C.SOLVER.LOSS = 'IoU'
__C.SOLVER.EPOCHS = 1
__C.SOLVER.OPTIMIZER = 'adam'
__C.SOLVER.ENERGY_FACTOR = 0.0001 
__C.SOLVER.EBM_WEIGHTS = "energy/torch_model.pth"


## Focal Loss parameters
__C.SOLVER.GAMMA = 2 
__C.SOLVER.ALPHA = 0.5 
__C.SOLVER.WEIGHTS = [1, 1] 
__C.SOLVER.TRAINABLE_WEIGHTS = False

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()


def combine_cfgs(path_cfg_data: Path=None, path_cfg_override: Path=None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data=Path(path_cfg_data)
        if not path_cfg_data.exists():
            print(f"[ERROR]: {path_cfg_data} doesn't exist.")
            sys.exit()
            
    if path_cfg_override is not None:
        path_cfg_override=Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # # Merge from .env
    # # Priority 1:
    # list_cfg = update_cfg_using_dotenv()
    # if list_cfg is not []:
    #     cfg_base.merge_from_list(list_cfg)

    return cfg_base


def main():
    cfg_path = "experiments/exp01.yaml"
    cfg  = combine_cfgs(cfg_path) 
    print(cfg)


if __name__ == "__main__":
    main()