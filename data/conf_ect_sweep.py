# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
   Generate Confocal & ECT datasets
"""

import os 
import math
import argparse 
import imageio
import cv2


try: 
    from utils import read_yaml, resize_cfg
    from confocal import read_confocal, get_column, get_depth_image, build_depth_image_2, conf_image_size, preprocess, plot_conf_images
    from minerva import read_ect, get_ect_data
    from plot import plot_confocal_ECT, draw_line, draw_grid, plot, sweep_frame, sweep_frame_2

    from minerva import simulate_biofilm
    from generate_data import ECTInstance
    from tree_generator import TreeGenerator
except Exception as e: 
    from .utils import read_yaml, resize_cfg
    from .confocal import read_confocal, get_column, get_depth_image, build_depth_image_2, conf_image_size, preprocess, plot_conf_images
    from .minerva import read_ect, get_ect_data
    from .plot import plot_confocal_ECT, draw_line, draw_grid, plot, sweep_frame, sweep_frame_2
    from .minerva import simulate_biofilm
    from .generate_data import ECTInstance
    from .tree_generator import TreeGenerator
    
    
index = 0 


def generate_data(slice_col, i, ect_images, row_offsets, z_stack, conf_image, ect_cfg, conf_cfg, output_dir, tree_generator, save_all):
    row_range = [i, i+ect_cfg.ROW_OFFSET]
    minerva_data = get_ect_data(ect_images, row_offsets, ect_cfg.MAX_ROW_OFFSET, ect_cfg.MIN_ROW_OFFSET, slice_col, row_range, [slice_col, slice_col+ect_cfg.COL_OFFSET], ect_cfg, output_dir)

    # get corresponding cross sectional image from  confocal 
    if not conf_cfg.RESIZE_STACK: 
        confocal_column = math.ceil((slice_col * 10) / conf_cfg.PIXEL_SIZE_XY)
        conf_step = math.ceil((i*10)/conf_cfg.PIXEL_SIZE_XY) + 1
        conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]
    else:
        confocal_column = slice_col
        conf_step = i*10
        conf_row_range = [conf_step, conf_step + conf_cfg.ROW_OFFSET]

    # # quit if the confocal range is above the confocal image
    # if conf_row_range[1] > conf_image.shape[0]: 
    #     break 
    
    column = get_column(z_stack, slice_col) 
    column_processed = preprocess(column, os.path.join(output_dir,  f"{i}_cross_section_processed_{slice_col}.png"))
    column_processed[column_processed == 255] = conf_cfg.FOREGROUND_PERM
    column_processed[column_processed == 0] = conf_cfg.BACKGROUND_PERM
    
    cross_section = get_depth_image(column, conf_row_range)
    cross_section_processed = get_depth_image(column_processed, conf_row_range)

    # save_path = None 
    # if save_all:       
    #     save_path = os.path.join(output_dir, f"{i}_cross_section_processed_{slice_col}.png")
    
    # cross_section_processed = preprocess(cross_section, save_path)
    
    scaled_data = minerva_data
    scaled_data[:, 2] = minerva_data[:, 2]* 1e15 * 0.1
    
    save_path = None 
    if save_all:       
        save_path = os.path.join(output_dir, f"frame_{i}.png")
    
    im = sweep_frame(slice_col, confocal_column, row_range, conf_row_range, ect_images[row_offsets.index(-1)], conf_image,  minerva_data[:, 2], scaled_data[:, 2], cross_section_processed, cross_section_processed, cross_section, save_path)
    
    save(scaled_data, cross_section_processed, tree_generator)
    return im 


def sweep(slice_col, stride, ect_images, row_offsets, z_stack, conf_image, ect_cfg, conf_cfg, output_dir, tree_generator, save_all):
    
    myframes = []
    num_rows = ect_images[0].shape[0] - ect_cfg.ROW_OFFSET
    
    myframes = [generate_data(slice_col, i, ect_images, row_offsets, z_stack, conf_image, ect_cfg, conf_cfg, output_dir, tree_generator, save_all) 
     for i in range(0, num_rows, stride)]
    
    print(f"Done {slice_col}", flush=True)
    print(index, flush=True)
    # create .mp4 video file
    imageio.mimsave(os.path.join(output_dir, f'frames_{slice_col}_2.mp4'), myframes, fps=5)
    
    # return myframes


def save(minerva_data, confocal_image, tree_generator): 
    global index 
    # save to their corresponding dir 
    ect_instance = ECTInstance(index, num_beads=0, perm=[], perm_xy=confocal_image, dperm=[], ext_mat=minerva_data[:,:2].astype(int))
    ect_instance.v_b = minerva_data[:,2] 
    ect_instance.write(tree_generator)

    ect_instance.plot(mesh_points=None, mesh_triangles=None, mesh_params=None, el_pos=None, output_tree=tree_generator, debug=True)
    tree_generator.write_json({f"{index}": ect_instance.dict(tree_generator.root_dir, is_exp=True)})

    index = index + 1 
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ect', type=str, help="Path to ECT .h5 file.", required=True)
    parser.add_argument('--confocal', type=str, help="Path to confocal .tiff file.", required=True)
    parser.add_argument('--ect_cfg', type=str, help="Path to ECT config. .yaml file.", required=True)
    parser.add_argument('--confocal_cfg', type=str, help="Path to confocal config. .yaml file.", required=True)
    parser.add_argument('--stride', type=int, help="value of stride", default=4)
    parser.add_argument('--simulate', type=bool, help="Run ECT simulations with the confocal as ground truth.", default=True)
    parser.add_argument('--save_all', type=bool, help="Save all files, useful for debugging", default=False)
    parser.add_argument('--num_threads', type=int, help="Number of threads", default=10)
    parser.add_argument('--output_dir', type=str, help="Path output directorty. ", required=True)

    args = parser.parse_args()

    ect_file = args.ect 
    confocal_file = args.confocal
    ect_cfg_file = args.ect_cfg
    confocal_cfg_file = args.confocal_cfg
    stride = args.stride
    simulate = args.simulate
    save_all = args.save_all
    num_threads = args.num_threads
    output_dir = args.output_dir
    
    conf_dir = os.path.join(output_dir, "Confocal")
    ect_dir = os.path.join(output_dir, "ECT")
    dataset_dir = os.path.join(output_dir, "dataset")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ect_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    tree_generator = TreeGenerator(root_dir=dataset_dir)
    tree_generator.generate() 

    # read yaml cfg 
    ect_cfg = read_yaml(ect_cfg_file)
    confocal_cfg = read_yaml(confocal_cfg_file)

    ### 1. Read confocal  ###
    conf_img_stack, conf_image, conf_maxZ = read_confocal(confocal_file, confocal_cfg, output_dir)
    print("Confocal shape: ", conf_image.shape, flush=True)
    
    conf_image_resized = cv2.resize(conf_image, dsize=(256, 512))

    plot_conf_images(conf_image, conf_maxZ, conf_image_resized, confocal_cfg, conf_dir)

    ### 2. Read ECT ###
    ect_images, row_offsets, col_offsets = read_ect(ect_file, ect_cfg, ect_dir)
    print("Row Offsets: ", row_offsets, flush=True)
   
    ## Plot ect v.s confocal ##
    save_path = os.path.join(output_dir, "conf_ect.png")
    conf_cfg_resized = resize_cfg(confocal_cfg, conf_image.shape, (512, 256))
    plot_confocal_ECT(conf_image_resized, ect_images[row_offsets.index(-1)], ect_cfg, conf_cfg_resized, save_path)

    ## 3. Sweep ECT and Confocal Image ##
    start_column = ect_cfg.START_COLUMN
    end_column = ect_cfg.END_COLUMN

    for slice_col in range(start_column, end_column):
        sweep(slice_col, stride, ect_images, row_offsets, conf_img_stack, conf_image, ect_cfg, confocal_cfg, output_dir, tree_generator, save_all) 
    

if __name__ == "__main__":
    main() 