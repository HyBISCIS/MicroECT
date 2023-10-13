
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  

import os 
import sys
import argparse 

import cv2 
import imageio
import numpy as np 
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import io
from skimage.transform import resize


try: 
    from .plot import plot_confocal, plot_box_annotations, draw_line, draw_grid, plot_corner_pts
    from .utils import read_yaml 
    from .generate_data import ECTInstance, create_mesh, create_ex_mat
    from .minerva import simulate_biofilm
    from .mesh_params import MeshParams
except Exception as e: 
    from plot import plot_confocal, plot_box_annotations, draw_line, draw_grid, plot_corner_pts
    from utils import read_yaml 
    from generate_data import ECTInstance, create_mesh, create_ex_mat
    from minerva import simulate_biofilm
    from mesh_params import MeshParams


def read_confocal(tiff_file, config, output_dir):
    im = io.imread(tiff_file) 
    print("Confocal Shape: ", im.shape)

    if len(im.shape) == 4: 
        z, x, y, _ = im.shape
    else: 
        z, x, y = im.shape

    img_stack = np.zeros((x,y,z))

    # Normalize image channels
    if len(im.shape) == 4: 
        head, _ = os.path.split(tiff_file)
        if not Path(os.path.join(head, 'img_stack.npy')).exists(): 
            for i in range(z):
                img_stack[:,:,i] = resize(np.linalg.norm(im[i, :, :, :], axis=2), (x, y))
            
            with open(os.path.join(head, 'img_stack.npy'), 'wb') as f:
                np.save(f, np.asarray(img_stack))
        else: 
            img_stack = np.load(os.path.join(head, 'img_stack.npy'))
    else:
        for i in range(z):
            img_stack[:, :, i] = im[i, :, :]

    if config.FLIP_Z: 
        img_stack = np.flip(img_stack, 2)

    # Crop outer pixels from confocal stack 
    img_stack = img_stack[config.CROP_Y[0]:config.CROP_Y[1], config.CROP_X[0]:config.CROP_X[1], :]

    # img_stack = wrap_transform(img_stack)
    if config.ROTATE_Z_STACK_180: 
        img_stack = img_stack[::-1,::-1]

    conf_image = np.max(img_stack[:,:,:], axis=2)
   
    if config.WRAP_TRANSFORM:       
        save_path = os.path.join(output_dir, "confocal_corner_pts.png")
        plot_corner_pts(conf_image, [config.POINT_A, config.POINT_B, config.POINT_C, config.POINT_D], "Corner Points", "Cols", "Rows", save_path)
        
        img_stack = wrap_transform(img_stack, config)
        conf_image = np.max(img_stack[:,:,:], axis=2)

        trans_image_size = conf_image_size(conf_image, config)
        print("Wrapped Shape: ", img_stack[:, :, 0].shape, "Size (microns)", trans_image_size[0], trans_image_size[1])

    if config.RESIZE_STACK: 
        img_stack = resize_z_stack(img_stack)

    # if config.FIX_TILT: 
    #     save_path = os.path.join(output_dir, "confocal_tilt_pts.png")
    #     print(img_stack.shape)
    #     print("img stack column index shape: ", img_stack[:, -1, :].shape)
    #     plot_corner_pts(img_stack[:, -1, :].reshape(z, img_stack.shape[0]), [config.TILT_POINT_A, config.TILT_POINT_B, config.TILT_POINT_C, config.TILT_POINT_D], "Corner Points", "Row (y)", "Depth (z)", save_path, aspect_ratio=5.5)

    #     img_stack = fix_tilt(img_stack, config)

    conf_image = np.max(img_stack[:,:,:], axis=2)
    conf_maxZ = np.argmax(img_stack[:,:,:], axis=2)
    # conf_maxZ = np.zeros(conf_image.shape).astype(np.uint8)

    return img_stack, conf_image, conf_maxZ 


def resize_z_stack(img_stack):
    x, y, z = img_stack.shape
    img_stack_sm = np.zeros((5120, 256, z)) # each pixel is 1um except for z-stack
    for idx in range(z):
        img = img_stack[:, :, idx]
        img_sm = cv2.resize(img, (256, 5120), interpolation=cv2.INTER_CUBIC)
        img_stack_sm[:, :, idx] = img_sm
    return img_stack_sm


def conf_image_size(confg_image, config):
    shape = confg_image.shape 
    size_microns = (shape[0] * config.PIXEL_SIZE_XY, shape[1] * config.PIXEL_SIZE_XY) 
    return size_microns 


def get_column(img_stack, slice_col):
    _, _, z = img_stack.shape
    # Build the column cross section from the z-stack
    column_cross_section = [img_stack[:, slice_col, z-j-1] for j in range(1, z+1)]
    column_cross_section = np.array(column_cross_section).reshape((z, -1))
    return column_cross_section


def get_depth_image(column, row_range):
    z,_ = column.shape
    cross_section = column[:, row_range[0]:row_range[1]]
    cross_section = np.array(cross_section).reshape((z, -1))
    return cross_section


def build_depth_image(img_stack, config, box_index):
    _, _, z = img_stack.shape
    row_range = config.ROWS[box_index]
    slice_col = config.COLUMNS[box_index]
    
    # Build the cross section from the z-stack
    cross_section = [img_stack[row_range[0]:row_range[1], slice_col, z-j-1] for j in range(1, z+1)]
    cross_section = np.array(cross_section).reshape((z, config.ROW_OFFSET))
    
    # normalize cross section with the pixel size --> returns the image in microns instead of pixels
    # reisze cross section 
    if not config.RESIZE_STACK: 
        dsize = (int(cross_section.shape[1]*config.PIXEL_SIZE_XY), int(cross_section.shape[0]*config.PIXEL_SIZE_Z))
        cross_section_resized = cv2.resize(cross_section, dsize=dsize, interpolation=cv2.INTER_CUBIC) 
    else: 
        cross_section_resized = cross_section
        
    # resize to desired size (microns, microns) 
    rows, cols = config.DSIZE
    
    # Crop the grid offset if it is in the config
    if len(config.GRID_OFFSET) > 0: 
        grid_offset = config.GRID_OFFSET[box_index]
        image = cross_section_resized[grid_offset:rows+grid_offset, 0:cols]
    else:
        image = cross_section_resized[0:rows, 0:cols]

    return image


def build_depth_image_2(img_stack, config, row_range, slice_col):
    _, _, z = img_stack.shape
    row_range = row_range
    slice_col = slice_col
    
    # Build the cross section from the z-stack
    cross_section = [img_stack[row_range[0]:row_range[1], slice_col, z-j-1] for j in range(1, z)]
    cross_section = np.array(cross_section).reshape((z-1, -1))

    # normalize cross section with the pixel size 
    dsize = (int(cross_section.shape[1]*config.PIXEL_SIZE_XY), int(cross_section.shape[0]*config.PIXEL_SIZE_Z))
    cross_section_resized = cv2.resize(cross_section, dsize=dsize, interpolation=cv2.INTER_CUBIC) 
    
    # resize to desired size 
    rows, cols = config.DSIZE
    image = cross_section_resized[0:rows, 0:cols]
   
    return cross_section_resized


def preprocess(image, save_path):
    # Erase dots from iamge 
    # 1. binarize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # blurred = cv2.GaussianBlur(image, (9, 9), 0)
    blurred = cv2.medianBlur(image, 13)

    binr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
    # binr = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    kernel = np.ones((3, 3),np.uint8)
    binr = cv2.dilate(binr, kernel, iterations = 2)

    # 2. Remove small dots from image
    # if row_offset >= 468: 
    #     print("erroding more")
    #     kernel = np.ones((12, 12),np.uint8)
    #     iterations = 5
    # else:
    kernel = np.ones((3, 3),np.uint8)
    iterations = 3
    erosion = cv2.erode(binr, kernel, iterations = iterations)
    
    kernel = np.ones((1, 1),np.uint8)
    iterations = 2
    erosion = cv2.erode(erosion, kernel, iterations = iterations)


    # 2. Medina blur to smooth out edges
    blur = cv2.medianBlur(erosion, 7)
    
    # # 3. Fill in gaps in the image
    # kernel = np.ones((1, 1),np.uint8)
    # dilation = cv2.dilate(erosion, kernel, iterations = 1)

    blur = blur.astype("float32")

    return blur 


def plot_conf_images(conf_image, conf_maxZ, conf_image_resized, confocal_cfg, output_dir):
    save_path = os.path.join(output_dir, "conf_image.png")
    plot_confocal(conf_image, f'Confocal Image {conf_image_size(conf_image, confocal_cfg)}u', f"Columns ({conf_image.shape[1]})", f"Rows ({conf_image.shape[0]})", save_path)
    
    save_path = os.path.join(output_dir, "conf_image_maxz.png")
    plot_confocal(conf_maxZ*1.5, 'Confocal Image (max)', "Columns", "Rows", save_path)
  
    save_path = os.path.join(output_dir, "confocal_boxes.png")
    plot_box_annotations(conf_image, confocal_cfg.COLUMNS, confocal_cfg.ROWS, confocal_cfg.COL_RANGE, [None, None], [None, None], "Box Annotations", "Columns", "Rows", save_path)    
    
    save_path = os.path.join(output_dir, "conf_image_resized.png")
    plot_confocal(conf_image_resized, "Confocal Image Resized (256, 512)", "Columns", "Rows", save_path)
 

def generate_video(z_stack, output_dir):
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
    frames = [(z_stack[:, :, i]).astype(np.uint8) for i in range(0, z_stack.shape[-1])]

    for i, frame in enumerate(frames): 
        plot_confocal(frame, f'Confocal Image {i}', "Columns", "Rows", os.path.join(output_dir, 'frames', f'frame_{i}.png'), cmap='Reds')

    # save_path = os.path.join(output_dir, f'confocal.mp4')
    # imageio.mimsave(save_path, frames, fps=1)


def wrap_transform(image_stack, config):
    # [cols, rows]
    pt_A  = config.POINT_A
    pt_B  = config.POINT_B
    pt_C  = config.POINT_C
    pt_D  = config.POINT_D

    # Here, I have used L2 norm. You can use L1 also.
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth  = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    print("Max Height: ", maxHeight, "Max Width: ", maxWidth)
    output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
    
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(np.float32(input_pts), np.float32(output_pts))
    # Transform the image
    transformed_image = cv2.warpPerspective(image_stack[:, :, 0], M, (maxWidth, maxHeight)) 
    x,y = transformed_image.shape

    img_stack_wrapped = np.zeros((x, y, image_stack.shape[2]))
    img_stack_wrapped[:, :, 0] = transformed_image
    
    for i in range(1, img_stack_wrapped.shape[2]):
        img_stack_wrapped[:, :, i] = cv2.warpPerspective(image_stack[:, :, i], M, (maxWidth, maxHeight)) 
        
    return img_stack_wrapped


def fix_column_tilt(column, config):
    # [cols, rows]
    pt_A = config.TILT_POINT_A
    pt_B = config.TILT_POINT_B
    pt_C = config.TILT_POINT_C
    pt_D = config.TILT_POINT_D

    # Here, I have used L2 norm. You can use L1 also.
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth  = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    print("Max Height: ", maxHeight, "Max Width: ", maxWidth)
    output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
    
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(np.float32(input_pts), np.float32(output_pts))
    # Transform the image
    transformed_image = cv2.warpPerspective(column, M, (maxWidth, maxHeight)) 
    
    return transformed_image


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--confocal', type=str, help="Path to merged .tif file.", required=False)
    parser.add_argument('--confocal_cfg', type=str, help="Path to config file .yaml file.", required=True)
    parser.add_argument('--simulate', type=bool, help="Simulate the biofilm in pyEIT.", default=False)
    parser.add_argument('--input_dir', type=str, help="Path to input directory of .tif series if merged file not available.", required=False)
    parser.add_argument('--output_dir', type=str, help="Path to dataset output directory.", required=True)

    args = parser.parse_args()

    input_file = args.confocal 
    input_dir = args.input_dir 
    yaml_cfg = args.confocal_cfg
    simulate = args.simulate
    output_dir = args.output_dir 

    os.makedirs(output_dir, exist_ok=True)

    if input_file is None and input_dir is None: 
        print("[ERROR]: Missing input_file/iput_dir arguments.")
        sys.exit(1)

    # Read YAML CFG file 
    config = read_yaml(yaml_cfg)

    # Read .tif files
    img_stack, conf_image, conf_maxZ = read_confocal(input_file, config, output_dir)
    image_size = conf_image_size(conf_image, config)
    
    print("confocal size: ", image_size)
    # Resize confocal image to align with ECT image 
    conf_image_resized = cv2.resize(conf_image, dsize=(256, 512))
    
    # Plot images
    plot_conf_images(conf_image, conf_maxZ, conf_image_resized, config, output_dir)

    # generate confocal video --> Helps with visualizing the tilt 
    generate_video(img_stack, output_dir)
    
    x, y, z = img_stack.shape
    for i, slice_col in enumerate(config.COLUMNS):
        column = get_column(img_stack, slice_col) 

        if config.FIX_TILT: 
            column_fixed = fix_column_tilt(column, config)
            save_path = os.path.join(output_dir,  f"{i}_column_fixed_{slice_col}.png")
            plot_confocal(column_fixed, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, ticks=True, scale_bar=False, save_path=save_path, aspect_ratio=5.5)
            
            save_path = os.path.join(output_dir,  f"{i}_column_corner_pts_{slice_col}.png")
            plot_corner_pts(column, [config.TILT_POINT_A, config.TILT_POINT_B, config.TILT_POINT_C, config.TILT_POINT_D], "Corner Points", "Row (y)", "Depth (z)", save_path, aspect_ratio=5.5)
            column = column_fixed
            
        column_processed = preprocess(column, os.path.join(output_dir,  f"{i}_cross_section_processed_{slice_col}.png"))
        
        column_processed[column_processed == 255] = config.FOREGROUND_PERM
        column_processed[column_processed == 0] = config.BACKGROUND_PERM

        save_path = os.path.join(output_dir,  f"{i}_column_processed_{slice_col}.png")
        plot_confocal(column_processed, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, ticks=True, scale_bar=False, save_path=save_path, aspect_ratio=5.5) 
       
        save_path = os.path.join(output_dir,  f"{i}_column_raw_{slice_col}.png")
        plot_confocal(column, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=12, ticks=True, scale_bar=False, save_path=save_path, aspect_ratio=5.5)

        cross_section = get_depth_image(column_processed, config.ROWS[i])
        save_path = os.path.join(output_dir,  f"{i}_cross_section_{slice_col}.png")

        plot_confocal(cross_section, "Confocal Microscopy", "Row (200\u03bcm)", "Depth (100\u03bcm)", font_size=24, ticks=True, scale_bar=False, save_path=save_path) 
        plot_confocal(cross_section, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=24, ticks=False, scale_bar=True, save_path=os.path.join(output_dir,  f"{i}_cross_section_scale_bar_{slice_col}.png")) 

        # build the cross-section without applying the filter across the whole column --> for reference
        cross_section = build_depth_image(img_stack, config, i)
        save_path = os.path.join(output_dir,  f"{i}_cross_section_raw_{slice_col}.png")
        
        plot_confocal(cross_section, "Confocal Microscopy", "Row (200\u03bcm)", "Depth (100\u03bcm)", font_size=24, ticks=True, scale_bar=False, save_path=save_path) 
        plot_confocal(cross_section, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=24, ticks=False, scale_bar=True, save_path=os.path.join(output_dir,  f"{i}_cross_section_scale_bar_raw_{slice_col}.png")) 

        binr = preprocess(cross_section, os.path.join(output_dir,  f"{i}_cross_section_processed_raw_{slice_col}.png"))

        binr[binr == 255] = config.FOREGROUND_PERM
        binr[binr == 0] = config.BACKGROUND_PERM
      
        plot_confocal(binr, "Confocal Microscopy", "Row (y)", "Depth (z)", font_size=24, ticks=False, scale_bar=True, save_path=os.path.join(output_dir,  f"{i}_cross_section_raw_scale_bar_processed_{slice_col}.png")) 

        if simulate: 
            ex_mat  = create_ex_mat(MeshParams)
            simulated_data, mesh_params, perm, perm_xy, dperm, pts, tri, el_pos = simulate_biofilm(binr, ex_mat, MeshParams)
        
            save_path = os.path.join(output_dir,  f"{i}_perm_{slice_col}.png")
            draw_grid(perm_xy[0], "Permittivity", save_path)        
            
            save_path = os.path.join(output_dir,  f"{i}_sim_{slice_col}.png")
            draw_line(np.arange(0, len(simulated_data[0][:, 2])), simulated_data[0][:, 2], "Simulated Data", "Measurement Index", "Simulated Data ", save_path )


if __name__ == "__main__":
    main()