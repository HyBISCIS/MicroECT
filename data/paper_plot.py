# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

# create plot for confocal image and ECT image
import os 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
from matplotlib_scalebar.scalebar import ScaleBar


import os 
import math
import argparse 
import imageio
import itertools
import cv2
import concurrent.futures
from concurrent.futures import wait

import numpy as np
from threading import Thread, Lock

from utils import read_yaml, resize_cfg
from confocal import read_confocal, build_depth_image_2, conf_image_size, preprocess, plot_conf_images
from minerva import read_ect, get_ect_data
from plot import plot_confocal_ECT, draw_line, draw_grid, plot, sweep_frame, sweep_frame_2

from minerva import simulate_biofilm
from generate_data import ECTInstance
from tree_generator import TreeGenerator


def plot_confocal(image, title, xlabel, ylabel, font_size, ticks, scale_bar, save_path, format='png'):
    plt.figure(figsize=(24,12))
    # plt.rcParams.update({'font.size': font_size})

    fig, ax1 = plt.subplots()

    ax1.imshow(image, vmin=(np.mean(image)-(2*np.std(image))), vmax=(np.mean(image)+(5*np.std(image))), cmap='Reds')
    # trans = ax1.get_xaxis_transform()
    # ax1.annotate('1.5u', xy=(1, -.1), xycoords=trans, ha="center", va="top")
    # ax1.plot([-.4,2.4],[-.08,-.08], color="k", transform=trans, clip_on=False)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylim([0, image.shape[0]])
    ax1.set_title(title, fontsize=font_size, color='r')
   
    if not ticks: 
        plt.tick_params(left = False, right = False , labelleft = False,labelbottom = False, bottom = False)
    
    if scale_bar:
        # add scale bar
        scalebar = ScaleBar(10, "um", length_fraction=0.05, width_fraction=0.04, color='k', box_alpha=1)
        plt.gca().add_artist(scalebar)
   
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(.1)  # change width
        ax1.spines[axis].set_color('k')    # change color

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.margins(x=0)
    plt.close()


def plot_ect(image, xrange, vrange, title, xlabel, ylabel, font_size, ticks, scale_bar, save_path, format='png'): 
    plt.figure(figsize=(12,12))
    # plt.rcParams.update({'font.size': font_size})

    fig, ax = plt.subplots()
    ymin,ymax=[0, image.shape[0]]
    xmin,xmax=xrange       
    vmin,vmax=vrange

    if vmin is None: 
        ax.imshow(image, vmin=(np.mean(image)-(2*np.std(image))), 
            vmax=(np.mean(image)+(12*np.std(image))), 
            cmap='viridis')
    else:
        ax.imshow(image,
                vmin=np.mean(image[ymin:ymax,xmin:xmax])+vmin*np.std(image[ymin:ymax,xmin:xmax]), 
                vmax=np.mean(image[ymin:ymax,xmin:xmax])+vmax*np.std(image[ymin:ymax,xmin:xmax]), 
                cmap='Blues')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)    
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.title(title, fontsize=font_size, color='b')
    
    if not ticks: 
        plt.tick_params(left = False, right = False , labelleft = False,labelbottom = False, bottom = False)
    
    if scale_bar:
        # add scale bar
        scalebar = ScaleBar(10, "um", length_fraction=0.5, width_fraction=0.018, box_color=None, color='w', location='lower left', label_loc='top', scale_loc='top', box_alpha=0)
        plt.gca().add_artist(scalebar)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(.1)  # change width
        ax.spines[axis].set_color('k')    # change color

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.margins(x=0)
    plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.close()



def main():
    output_dir = "logs/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    conf_path = "real/Larkin_Lab_Data/ECT_Train_07082022/Confocal/Region_1_Merged_RAW_ch00.tif"
    ect_path = "real/Larkin_Lab_Data/ECT_Train_07082022/Pre_Confocal_ECT_1D_Scans/ECT_scan_1D_07082022_set_0.h5"
    
    ect_cfg_file = "real/07082022_ECT.yaml"
    confocal_cfg_file = "real/07082022_Confocal.yaml"

    ect_cfg = read_yaml(ect_cfg_file)
    confocal_cfg = read_yaml(confocal_cfg_file)

    conf_img_stack, conf_image, conf_maxZ = read_confocal(conf_path, confocal_cfg)
    # conf_maxZ = np.argmax(conf_img_stack[:,:,:], axis=2)
    print("Confocal shape: ", conf_image.shape)

    conf_image_resized = cv2.resize(conf_image, dsize=(256, 512))
    plot_conf_images(conf_image, conf_maxZ, conf_image_resized, confocal_cfg, output_dir)

    plot_confocal(conf_image, "Confocal Image", "Column (2560\u03bcm)", "Row (5120\u03bcm)", font_size=12, ticks=True, scale_bar=False, save_path=os.path.join(output_dir, "conf_image_plot.png"))
   
    plot_confocal(conf_image_resized, "Confocal Image", "Column (2560\u03bcm)", "Row (5120\u03bcm)", font_size=12, ticks=True, scale_bar=False, save_path=os.path.join(output_dir, "conf_image_plot2.png"))
    plot_confocal(conf_image_resized, "Confocal Image", "Column (2560\u03bcm)", "Row (5120\u03bcm)", font_size=12, ticks=True, scale_bar=False, save_path=os.path.join(output_dir, "conf_image_plot2.pdf"), format="pdf")

    plot_confocal(conf_image_resized, "Confocal Image", "", "", font_size=14, ticks=False, scale_bar=False, save_path=os.path.join(output_dir, "conf_image_plot2_scalebar.png"))
    plot_confocal(conf_image_resized, "Confocal Image", "", "", font_size=14, ticks=False, scale_bar=False, save_path=os.path.join(output_dir, "conf_image_plot2_scalebar.pdf"), format="pdf")

    # 2. Read ECT
    ect_images, row_offsets, col_offsets = read_ect(ect_path, ect_cfg, output_dir)
    print("Row Offsets: ", row_offsets)
   
    plot_ect(ect_images[row_offsets.index(-1)], xrange=[100,450], vrange=[-4,1], title=f"ECT Image", xlabel="Column (2560\u03bcm)", ylabel="Row (5120\u03bcm)", font_size=12,  ticks=True, scale_bar=False, save_path=os.path.join(output_dir, "ect_image_plot.png"))
    plot_ect(ect_images[row_offsets.index(-1)], xrange=[100,450], vrange=[-4,1], title=f"ECT Image", xlabel="Column (2560\u03bcm)", ylabel="Row (5120\u03bcm)", font_size=12,  ticks=True, scale_bar=False, save_path=os.path.join(output_dir, "ect_image_plot.pdf"), format="pdf")

    plot_ect(ect_images[row_offsets.index(-1)], xrange=[100,450], vrange=[-4,1], title=f"ECT Image", xlabel="", ylabel="", font_size=14,  ticks=False, scale_bar=True, save_path=os.path.join(output_dir, "ect_image_plot_scalebar.png"))
    plot_ect(ect_images[row_offsets.index(-1)], xrange=[100,450], vrange=[-4,1], title=f"ECT Image", xlabel="", ylabel="", font_size=14,  ticks=False, scale_bar=True, save_path=os.path.join(output_dir, "ect_image_plot_scalebar.pdf"), format="pdf")


if __name__=="__main__":
    main()