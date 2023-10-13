
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os
import sys
import cv2
import glob
import json 
import copy 
import random 
random.seed(0)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as T

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, median_filter

try:
    from mesh_params import MeshParams
    from plot import draw_grid, draw_line, draw_boundary_measurement 
except:
    from .mesh_params import MeshParams
    from .plot import draw_grid, draw_line, draw_boundary_measurement 


class xData():
    
    def __init__(self, perm, perm_xy, dperm):
        self.perm = perm 
        self.perm_xy = perm_xy
        self.dperm = dperm 

    def load(self):
        perm = np.load(self.perm) 
        perm_xy = np.load(self.perm_xy)
        dperm = np.load(self.dperm)

        perm_xy[np.isnan(perm_xy)] = 1.0
        dperm[np.isnan(dperm)] = 0 

        perm_xy = perm_xy.reshape(1, perm_xy.shape[0], perm_xy.shape[1])

        return {"perm": perm, "perm_xy": perm_xy, "dperm": dperm}


class yData():

    def __init__(self, v_b):

        self.v_b = v_b  

        self.u = [] 
        self.u_xy = [] 
        self.du =  []

        self.ext_mat = [] 
        self.ext_elec_pos = [] 

    def add_solution(self, u, u_xy, du, ext_mat, ext_elec_pos):
        self.u.append(u)
        self.u_xy.append(u_xy)
        self.du.append(du)
        self.ext_mat.append(ext_mat)
        self.ext_elec_pos.append(ext_elec_pos)

    def load(self):
        u = []
        u_xy = []
        du = []

        num_excitations = len(self.ext_mat)
        for i in range(num_excitations):
               
            u.append(np.load(self.u[i])) 
            u_xy.append(np.load(self.u_xy[i]))
            du.append(np.load(self.du[i]))
            
            u_xy[i][np.isnan(u_xy[i])] = 0

        v_b = np.load(self.v_b)

        return {"u": u, "u_xy": u_xy, "du": du, "v_b": v_b, "ext_mat": self.ext_mat, "ext_elec_pos": self.ext_elec_pos} 


class Dataset(torch.utils.data.IterableDataset):

    def __init__(self, path, shuffle=False, normalize=False, standardize=False, smooth=False, pos_value=-1, neg_value=1, train_min=None, train_max=None, device='cuda'):
        """
            args: 
                shuffle: randomize the order of datapoints
                normalize: scale image pixels to be between [0, 1]
                standardize: scale image pixels to have zero mena and unit varience
        """
        if not os.path.exists(path):
            print(f"{path} doesn't exist.")
            sys.exit(1)

        self.path = path
        self.shuffle = shuffle 
        self.normalize = normalize  
        self.standardize = standardize
        self.smooth = smooth
        self.device = device 
        self.pos_value = pos_value
        self.neg_value = neg_value
        self.train_min = train_min
        self.train_max = train_max 
        
        self.json_path = os.path.join(self.path, f"dataset.json")
        
        if self.standardize or self.normalize: 
            self.vb_mean, self.vb_std, self.vb_min, self.vb_max = self.calc_vb_stats()
            self.img_mean, self.img_std, self.img_min, self.img_max = self.calc_image_stats()

        self.xData = []
        self.yData = []
        self.parse_data()

    def calc_image_stats(self):
        # Get list of all images in training directory
        perm_path = os.path.join(os.path.join(self.path, f"x/perm_xy/*.npy"))
        file_list = sorted(glob.glob(str(perm_path)))

        images = []
        for file in file_list:
            img = np.load(file)
            images.append(img)

        mean = np.mean(images, axis=0)
        std = np.std(images, axis=0)

        min = np.min(images)
        max = np.max(images)

        return mean, std, min, max

    def calc_vb_stats(self):
        # Get list of all images in training directory
        vb_path = os.path.join(os.path.join(self.path, f"y/v_b/*.npy"))
        file_list = sorted(glob.glob(vb_path))
        
        vb = []
        for file in file_list:
            measurement = np.load(file)
            vb.append(measurement)

        mean = np.mean(vb, axis=0)
        std = np.std(vb, axis=0)
        
        if self.train_min is None: 
            min =  np.min(vb)
            self.train_min = min
        else: 
            min = self.train_min 
        
        if self.train_max is None: 
            max = np.max(vb)
            self.train_max = max
        else:
            max = self.train_max 
            
        print(min)
        print(max)
        
        return mean, std, min, max 


    def gaussian(input, is_training, stddev=0.2):
        if is_training:
            return input + Variable(torch.randn(input.size()).cuda() * stddev)
        return input


    def preprocess(self, img, vb):
        img_normalized = copy.copy(img)
        vb_normalized  = vb
        
        img_normalized[img == 1] = self.neg_value
        img_normalized[img != 1] = self.pos_value
        
        # img_normalized = img
        # img_normalized[img != 1] = -1

        if self.normalize: 
            # img_normalized =  (img - self.img_min) / (self.img_max - self.img_min)
            vb_normalized =  (vb - self.vb_min) / (self.vb_max - self.vb_min)
        
        if self.standardize: 
            img_normalized = img - self.img_mean / self.img_std 
            vb_normalized = vb - self.vb_mean / self.vb_std 

        return img_normalized, vb_normalized 

    def parse_data(self):
        self.perm = []
        self.capacitence = []
        self.ext_mat = []
        self.number_lines = []
        self.num_beads = []
        
        f = open(self.json_path)
        data = json.load(f)

        for _, subdict in data.items():
            perm_path = os.path.join(self.path, subdict["perm"])
            perm_xy_path = os.path.join(self.path, subdict["perm_xy"]) 
            dperm_path = os.path.join(self.path, subdict["dperm"]) 

            xdata = xData(perm_path, perm_xy_path , dperm_path)
            self.xData.append(xdata)

            num_beads = subdict["num_beads"]
            self.num_beads.append(num_beads) 
            
            v_b = os.path.join(self.path, subdict["v_b"]) 

            # read FEM solutions
            solutions = subdict["u"]

            y = yData(v_b)
            for solution in solutions.items():
                _, solution = solution
                ext_elec = solution["ext_elec"]
                ext_elec_pos = solution["ext_elec_pos"]
                u = os.path.join(self.path, solution["u"])  
                u_xy = os.path.join(self.path, solution["u_xy"]) 
                du = os.path.join(self.path, solution["du"])  

                y.add_solution(u, u_xy, du, ext_elec, ext_elec_pos)

            self.yData.append(y)

        if self.shuffle: 
            c = list(zip(self.xData, self.yData))
            random.shuffle(c)
            self.xData, self.yData = zip(*c)

        self.num_datapoints = len(self.xData)

    def __iter__(self):
        for xdata, ydata in zip(self.xData, self.yData):
            x = xdata.load()
            y = ydata.load()

            # preprocess 
            x["perm_xy"], y["v_b"] = self.preprocess(x["perm_xy"], y["v_b"])
                    
            if self.smooth: 
                x["perm_xy"] = self.smooth_edges(x["perm_xy"])
            
            x["perm_xy"] = x["perm_xy"][:, 0:100, 0:200]

            yield x, y
    
    def smooth_edges(self, perm_xy, sigma=1):
        perm_smoothed = median_filter(perm_xy, size=5)
        return perm_smoothed
    
    def __getitem__(self, idx):
        xdata = self.xData[idx]
        ydata = self.yData[idx]

        x = xdata.load()
        y = ydata.load()
        
        x["perm_xy"], y["v_b"] = self.preprocess(x["perm_xy"], y["v_b"])
        
        if self.smooth: 
            x["perm_xy"] = self.smooth_edges(x["perm_xy"])

        x["perm_xy"] = x["perm_xy"][:, 0:100, 0:200]

        return x, y 

    def __len__(self):
        return self.num_datapoints


def main(): 
    batch_size = 1
   
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)

    root_dir = "synthetic"
    data_path = os.path.join(root_dir, "data/dataset-small-beads-final")

    dataset = Dataset(data_path, shuffle=False, normalize=True, smooth=True, standardize=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    for x, y in data_loader:
        num_ext = len(y["ext_mat"])
        print("Number of excitations: ", num_ext)

        draw_grid(x["perm_xy"][0][0], "permittivity", "", "", os.path.join(debug_dir, "perm_smoothed.png"))
        draw_line(np.arange(0, len(y["v_b"][0])), y["v_b"][0], "Boundary Measurement", "Measurement", "Capacitence Value", os.path.join(debug_dir, "vb.png"))
        # draw_boundary_measurement(y["v_b"][0], debug_dir)
        for i in range(num_ext):
            ext_elec = [y["ext_mat"][i][0].numpy(), y["ext_mat"][i][1].numpy()]    
            du_mag = np.sqrt(y["du"][i][0][0]**2 + y["du"][i][0][1]**2)    
            draw_grid(y["u_xy"][i][0], "electric potential", "", "", save_path=os.path.join(debug_dir, f"u_xy_{i}__ext_mat_{ext_elec[0]}_{ext_elec[1]}.png"))
            draw_grid(du_mag,  "electric_field", "", "", save_path=os.path.join(debug_dir, f"du_{i}__ext_mat_{ext_elec[0]}_{ext_elec[1]}.png"))
        break 


if __name__ == "__main__":
    main()