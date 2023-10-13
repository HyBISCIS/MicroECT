
 
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  


import yaml 
import numpy as np

class YamlCFG(object):
      
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def add_attr(self, key, val):
        setattr(self, key, val)


def read_yaml(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    yaml_cfg = YamlCFG(data_loaded)

    COLUMNS = [] 
    ROWS = []
    POSITION = []
    GRID_OFFSET = []
    for box in data_loaded["BOXES"]:
        COLUMNS.append(box['COLUMN'])
        ROWS.append([box["ROW"], box["ROW"]+yaml_cfg.ROW_OFFSET])  
        if "X" in box: 
            POSITION.append([box["X"], box["Y"]])

        if "GRID_OFFSET" in box: 
            GRID_OFFSET.append(box["GRID_OFFSET"])
            
    COL_RANGE = [[col-yaml_cfg.COL_OFFSET, col+yaml_cfg.COL_OFFSET] for col in COLUMNS]

    yaml_cfg.add_attr("ROWS", ROWS)
    yaml_cfg.add_attr("COL_RANGE", COL_RANGE)
    yaml_cfg.add_attr("COLUMNS", COLUMNS)
    yaml_cfg.add_attr("POSITION", POSITION)
    yaml_cfg.add_attr("GRID_OFFSET", GRID_OFFSET)

    return yaml_cfg

def resize_cfg(config, size, new_size):
    resized_cfg = {}
    resized_cfg["COLUMNS"] = [col * new_size[1] / size[1] for col in config.COLUMNS] 
    resized_cfg["ROW_OFFSET"] = config.ROW_OFFSET * new_size[0] / size[0]
    resized_cfg["COL_OFFSET"] = config.COL_OFFSET * new_size[1] / size[1]
    resized_cfg["ROWS"] = [[row[0]*new_size[0] / size[0], row[1]*new_size[0] / size[0]]  for row in  config.ROWS]
    resized_cfg["COL_RANGE"] = [ [range[0]* new_size[1] / size[1], range[1]* new_size[1] / size[1]]  for range in config.COL_RANGE]

    new_cfg = YamlCFG(resized_cfg)
    return new_cfg  

