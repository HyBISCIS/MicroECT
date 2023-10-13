
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

"""
   Align Confocal & ECT datasets
"""

import os 
import argparse 

import cv2
import numpy as np

from sklearn.linear_model import LinearRegression

try: 
    from utils import read_yaml, resize_cfg
    from confocal import read_confocal, build_depth_image, conf_image_size, preprocess, plot_conf_images
    from minerva import read_ect, get_ect_data
    from plot import plot_confocal, plot_box_annotations, plot_confocal_ECT, draw_line, draw_grid, plot
    from generate_data import ECTInstance, create_mesh, create_ex_mat
    from minerva import simulate_biofilm
    from mesh_params import MeshParams
    from tree_generator import TreeGenerator
except Exception as e: 
    from .utils import read_yaml, resize_cfg
    from .confocal import read_confocal, build_depth_image, conf_image_size, preprocess, plot_conf_images
    from .minerva import read_ect, get_ect_data
    from .plot import plot_confocal, plot_box_annotations, plot_confocal_ECT, draw_line, draw_grid, plot, draw_confocal_grid
    from .generate_data import ECTInstance, create_mesh, create_ex_mat
    from .minerva import simulate_biofilm
    from .mesh_params import MeshParams
    from .tree_generator import TreeGenerator
    
    
def run_simulatation(confocal_cross_sections, minerva_data, row_offsets, ect_cfg, output_dir):
    # Generate output directories
    index = 0
    tree_generator = TreeGenerator(root_dir=output_dir)
    tree_generator.generate() 

    ex_mat = minerva_data[0][:,:2].astype(int)

    for i, img in enumerate(confocal_cross_sections): 
        slice_col = ect_cfg.COLUMNS[i]
       
        log_path = os.path.join(tree_generator.exp_dir, f"{i} Column {slice_col}")
        os.makedirs(log_path, exist_ok=True)

        binr = preprocess(img, os.path.join(tree_generator.exp_dir, f"{i}_cross_section_processed_{slice_col}.png"))
        
        binr[binr == 255] = 0.25
        binr[binr == 0] = 1 

        # ex_mat  = create_ex_mat(MeshParams)
        simulated_data, mesh_params, perm, perm_xy, dperm, pts, tri, el_pos = simulate_biofilm(binr, ex_mat, MeshParams)
    
        save_path = os.path.join(log_path,  f"{i}_perm_{slice_col}.png")
        draw_grid(perm_xy, "Permittivity", save_path)        
        
        save_path = os.path.join(log_path,  f"{i}_sim_{slice_col}.png")
        draw_line(np.arange(0, len(simulated_data[:, 2])), simulated_data[:, 2], "Simulated Data", "Measurement Index", "Simulated Data ", save_path )
        
        save_path = os.path.join(log_path, f"{i}_{slice_col}__raw_data.png")
        draw_line(np.arange(0, len(minerva_data[i][:, 2])), minerva_data[i][:, 2], "Raw Minerva data", "Measurement Index", "Minerva Data", save_path)

        plot(simulated_data, minerva_data[i], row_offsets, os.path.join(log_path, f"{i}__sim_real_pre_scaling.png"))

        for spacing in row_offsets:
            pats = np.where(np.abs(simulated_data[:,1]-simulated_data[:,0])==np.abs(spacing))[0]
            if(len(pats)>0):
                sim_data = simulated_data[pats,2]
                measured_data = minerva_data[i][pats,2]

                alpha = LinearRegression().fit(measured_data.reshape(-1, 1), sim_data.reshape(-1, 1))
                print(alpha.coef_, " ", alpha.intercept_)
                minerva_data[i][pats, 2] = measured_data * alpha.coef_ + alpha.intercept_

        plot(simulated_data, minerva_data[i], row_offsets, os.path.join(log_path, f"{i}__sim_real_post_scaling.png"))
        
        save_path = os.path.join(log_path,  f"{i}_meas_{slice_col}.png")
        draw_line(np.arange(0, len(minerva_data[i][:, 2])), minerva_data[i][:, 2], "Measured Data", "Measurement Index", "Measured Data ", save_path )

        index = save(index, simulated_data, minerva_data[i], perm, perm_xy, dperm, pts, tri, mesh_params, el_pos, tree_generator)

    return simulated_data, minerva_data


def save(index, simulated_data, minerva_data, perm, perm_xy, dperm, pts, tri, mesh_params, el_pos, tree_generator):

    # save to their corresponding dir 
    ect_instance = ECTInstance(index, num_beads=0, perm=perm, perm_xy=perm_xy, dperm=dperm, ext_mat=minerva_data[:,:2].astype(int))
    ect_instance.v_b = minerva_data[:,2] 
    ect_instance.write(tree_generator)
    ect_instance.plot(pts, tri, mesh_params, el_pos, tree_generator, True)
    tree_generator.write_json({f"{index}": ect_instance.dict(tree_generator.root_dir, is_exp=True)})


    # save to their corresponding dir
    index= index + 1 
    ect_instance = ECTInstance(index, num_beads=0, perm=perm, perm_xy=perm_xy, dperm=dperm, ext_mat=minerva_data[:,:2].astype(int))
    ect_instance.v_b = simulated_data[:,2] 
    ect_instance.write(tree_generator)
    ect_instance.plot(pts, tri, mesh_params, el_pos, tree_generator, True)
    tree_generator.write_json({f"{index}": ect_instance.dict(tree_generator.root_dir)})
    
    index = index + 1 

    return index 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ect', type=str, help="Path to ECT .h5 file.", required=True)
    parser.add_argument('--confocal', type=str, help="Path to confocal .tiff file.", required=True)
    parser.add_argument('--ect_cfg', type=str, help="Path to ECT config. .yaml file.", required=True)
    parser.add_argument('--confocal_cfg', type=str, help="Path to confocal config. .yaml file.", required=True)
    parser.add_argument('--simulate', type=bool, help="Run ECT simulations with the confocal as ground truth.", default=False)
    parser.add_argument('--output_dir', type=str, help="Path output directorty. ", required=True)

    args = parser.parse_args()

    ect_file = args.ect 
    confocal_file = args.confocal
    ect_cfg_file = args.ect_cfg
    confocal_cfg_file = args.confocal_cfg
    simulate = args.simulate
    output_dir = args.output_dir
    
    conf_dir = os.path.join(output_dir, "Confocal")
    ect_dir = os.path.join(output_dir, "ECT")
    dataset_dir = os.path.join(output_dir, "dataset")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ect_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # read yaml cfg 
    ect_cfg = read_yaml(ect_cfg_file)
    confocal_cfg = read_yaml(confocal_cfg_file)

    ### 1. Read confocal  ###
    conf_img_stack, conf_image, conf_maxZ = read_confocal(confocal_file, confocal_cfg, output_dir)
    conf_image_resized = cv2.resize(conf_image, dsize=(256, 512))

    plot_conf_images(conf_image, conf_maxZ, conf_image_resized, confocal_cfg, output_dir)

    confocal_cross_sections = []
    for i, slice_col in enumerate(confocal_cfg.COLUMNS): 
        cross_section = build_depth_image(conf_img_stack, confocal_cfg, i)
        confocal_cross_sections.append(cross_section)
        save_path = os.path.join(conf_dir, f"{i}_cross_section_{slice_col}.png")
        # plot_confocal(cross_section, "Cross-sectional Image", "x", "Depth (z) in microns", save_path) 
        plot_confocal(cross_section, "Ground Truth",  f"Row ({confocal_cfg.DSIZE[0]}µ)", f"Depth ({confocal_cfg.DSIZE[0]}µ)", save_path=save_path, colorbar=False,  font_size=18)

    ### 2. Read ECT ###
    ect_images, row_offsets, col_offsets = read_ect(ect_file, ect_cfg, ect_dir)
   
    save_path = os.path.join(ect_dir, "ect_boxes.png")
    plot_box_annotations(ect_images[row_offsets.index(-1)], ect_cfg.COLUMNS, ect_cfg.ROWS, ect_cfg.COL_RANGE, [None, None], [None, None], "Box Annotations", "Columns", "Rows", save_path)    

    ## Plot ect v.s confocal ##
    save_path = os.path.join(output_dir, "conf_ect.png")
    conf_cfg_resized = resize_cfg(confocal_cfg, conf_image.shape, (512, 256))
    plot_confocal_ECT(conf_image_resized, ect_images[row_offsets.index(-1)], ect_cfg, conf_cfg_resized, save_path)

    minerva_data = []
    for i, col in enumerate(ect_cfg.COLUMNS): 
        col_dir = os.path.join(ect_dir, f"{i}_Column_{col}")
        os.makedirs(col_dir, exist_ok=True)
        data = get_ect_data(ect_images, row_offsets, ect_cfg.MAX_ROW_OFFSET, ect_cfg.MIN_ROW_OFFSET , col, ect_cfg.ROWS[i], ect_cfg.COL_RANGE[i], ect_cfg, col_dir)
        minerva_data.append(data)

        save_path = os.path.join(col_dir, f"{i}_{col}__raw_data.png")
        print(save_path)

        draw_line(np.arange(0, len(data[:, 2])), data[:, 2], "Raw Minerva data", "Measurement Index", "Minerva Data ", save_path )

    ## 3. Simulate with the confocal as the ground truth ## 
    if simulate: 
       simulated_data, minerva_data = run_simulatation(confocal_cross_sections, minerva_data, row_offsets, ect_cfg, dataset_dir)



if __name__ == "__main__":
    main() 