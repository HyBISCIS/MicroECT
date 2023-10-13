
# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.  


import os 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
from matplotlib_scalebar.scalebar import ScaleBar

try: 
    from mesh_params import get_2d_grid, MeshParams
except ModuleNotFoundError: 
    from .mesh_params import get_2d_grid, MeshParams 


def plot(sim_data, real_data, row_offsets, save_path): 
    plt.figure(figsize=(14,7))    
    
    row_offsets = np.abs(row_offsets)
    for spacing in row_offsets:
        pats = np.where(np.abs(sim_data[:,1]-sim_data[:,0]) == spacing)[0]
        if(len(pats)>0):
            toplot = sim_data[pats,2]
            plt.subplot(1,2,1)
            plt.xlabel("Measurement")
            plt.ylabel("Capacitence (fF)")
            plt.plot(spacing + toplot,'.-', label=f"Row Offset {spacing}")
            plt.legend(loc="upper right")
            plt.title('simulated')

            toplot = real_data[pats,2] 
            toplot = toplot - np.min(toplot)
            toplot = toplot / np.max(toplot) * (np.max(sim_data[pats,2])-np.min(sim_data[pats,2]))
            toplot = toplot + np.min(sim_data[pats,2])
            plt.subplot(1,2,2)
            plt.xlabel("Measurement")
            plt.ylabel("Capacitence (fF)")
            plt.plot(spacing + toplot,'.-', label=f"Row Offset {spacing}")        
            plt.legend(loc="upper right")
            plt.title('minerva, measured & rescaled')
    
    plt.savefig(save_path)
    plt.close()



def plot_image_slice(image, slice_col, row_range, col_range, title, save_path):    
    fig, ax = plt.subplots()
    ax.imshow(image,  
            # vmin=(np.mean(image)-(2*np.std(image))), 
            # vmax=(np.mean(image)+(12*np.std(image))),
            cmap='Blues')

    anchor_point = (col_range[0], row_range[0])
    width, height = [col_range[1]-col_range[0]+1, row_range[1]-row_range[0]+1]

    rect = patches.Rectangle(anchor_point, width, height, linewidth=0.5, edgecolor='b', facecolor='none')

    ax.add_patch(rect)
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.title(title)
    plt.plot([slice_col, slice_col], [row_range[0], row_range[1]], linewidth=0.3, color="b")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_slice(image, slice_col, row_range, col_range, title, save_path):
    row_min, row_max = row_range 
    col_min, col_max = col_range

    fig = plt.figure(figsize=(12,9))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(grid[:3, :])
    im1 = ax.imshow(image, 
                    vmin=np.mean(image[row_min:row_max, col_min:col_max])-6*np.std(image[row_min:row_max, col_min:col_max]), 
                    vmax=np.mean(image[row_min:row_max, col_min:col_max])+1*np.std(image[row_min:row_max, col_min:col_max]), 
                    cmap='Blues')
    
    ax.set_title(title)
    
    plt.ylim([row_min, row_max])
    plt.xlim([col_min, col_max])
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.plot([slice_col, slice_col], [0, image.shape[0]], 'r')

    myslice_x = range(row_min, row_max)
    myslice_y = image[row_min:row_max, slice_col]
    
    # myslice_y = myslice_y - myslice_y[-1]

    ax = fig.add_subplot(grid[-1,:])                        
    im1 = ax.plot(myslice_x, myslice_y, 'r')
    plt.xlim([row_min, row_max])
    plt.xlabel("Rows")
    plt.ylabel("Capacitence Value")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def plot_image_slice_2(image, slice_col, xrange, yrange, title, save_path):    
    fig, ax = plt.subplots()
    ax.imshow(image,  
            vmin=(np.mean(image)-(2*np.std(image))), 
            vmax=(np.mean(image)+(12*np.std(image)))
            )

    anchor_point = (xrange[0], yrange[0])
    width, height = [xrange[1]-xrange[0]+1, yrange[1]-yrange[0]+1]

    rect = patches.Rectangle(anchor_point, width, height, linewidth=0.5, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.title(title)
    plt.plot([xrange[0], xrange[1]], [slice_col, slice_col], linewidth=0.3, color="r")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_slice_2(image, slice_col, xrange, yrange, title, save_path):
    xmin, xmax = xrange 
    ymin, ymax = yrange

    fig = plt.figure(figsize=(12,9))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(grid[:3, :])
    im1 = ax.imshow(image, 
                    vmin=np.mean(image[ymin:ymax,xmin:xmax])-6*np.std(image[ymin:ymax,xmin:xmax]), 
                    vmax=np.mean(image[ymin:ymax,xmin:xmax])+1*np.std(image[ymin:ymax,xmin:xmax]), 
                    cmap='Blues')
    
    ax.set_title(title)
    
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.plot([0, 512],[slice_col, slice_col],'r')

    myslice_x = range(xmin,xmax)
    myslice_y = image[slice_col, xmin:xmax]
    
    myslice_y = myslice_y - myslice_y[-1]

    ax = fig.add_subplot(grid[-1,:])                        
    im1 = ax.plot(myslice_x, myslice_y, 'r')
    plt.xlim([xmin, xmax])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_box_annotations(image, columns, row_range, col_range, xrange, vrange, title, xlabel, ylabel, save_path): 
    num_boxes = len(columns)

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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.title(title)

    for i in range(num_boxes):
        anchor_point = (col_range[i][0], row_range[i][0])
        width, height = [col_range[i][1]-col_range[i][0]+1, row_range[i][1]-row_range[i][0]+1]

        rect = patches.Rectangle(anchor_point, width, height, linewidth=0.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect) 
        ax.annotate(f"{i}", anchor_point, color='red', weight='normal', fontsize=6, ha='center', va='bottom')

        plt.plot([columns[i], columns[i]], [row_range[i][0], row_range[i][1]], linewidth=0.3, color="r")
    
    plt.savefig(save_path,  bbox_inches='tight')
    plt.close()


def plot_confocal(image, title, xlabel, ylabel, save_path, colorbar=False, scale_bar=False, ticks=True,  font_size=18, aspect_ratio=1, figsize=(6,6), cmap='Reds'):
    matplotlib.rcParams.update({'font.size': font_size})
    plt.figure(figsize=figsize)
    plt.imshow(image, 
            # vmin=(np.mean(image)-(2*np.std(image))), 
            # vmax=(np.mean(image)+(12*np.std(image))), 
            cmap=cmap)

    if colorbar:
        plt.colorbar()
    
    # add scale bar
    if scale_bar: 
        scalebar = ScaleBar(1, "um", length_fraction=0.26, width_fraction=0.04, color='0.2', location='upper right', label_loc='top', scale_loc='top', box_alpha=0)
        plt.gca().add_artist(scalebar)  
    
    if not ticks: 
        plt.tick_params(left = False, right = False , labelleft=False,labelbottom = False, bottom = False)

    plt.gca().set_aspect(aspect_ratio)
    plt.xlabel(xlabel, fontdict={'fontsize':font_size})
    plt.ylabel(ylabel, fontdict={'fontsize':font_size})
    plt.title(title, fontdict={'fontsize':font_size})
    plt.ylim([0, image.shape[0]])
    plt.savefig(save_path, bbox_inches='tight')
    plt.cla()
    plt.close()


def plot_corner_pts(image, pts, title, xlabel, ylabel, save_path, font_size=18, aspect_ratio=1, figsize=(6,6), cmap='Reds'):
    matplotlib.rcParams.update({'font.size': font_size})
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.tick_params(left = False, right = False , labelleft=False,labelbottom = False, bottom = False)
    plt.ylim([0, image.shape[0]])

    colors = ["red", "yellow", "green", "blue"]
    for i, pt in enumerate(pts): 
        plt.plot(pt[0], pt[1], marker='o', color=colors[i], markersize=1)

    # draw line between corner points
    pt_A = pts[0]
    pt_B = pts[1]
    pt_C = pts[2]
    pt_D = pts[3]
    plt.plot([pt_A[0], pt_B[0]], [pt_A[1], pt_B[1]], color="green", linewidth=0.5)
    plt.plot([pt_A[0], pt_D[0]], [pt_A[1], pt_D[1]], color="green", linewidth=0.5)
    plt.plot([pt_B[0], pt_C[0]], [pt_B[1], pt_C[1]], color="green", linewidth=0.5)
    plt.plot([pt_C[0], pt_D[0]], [pt_C[1], pt_D[1]], color="green", linewidth=0.5)

    plt.gca().set_aspect(aspect_ratio)
    plt.xlabel(xlabel, fontdict={'fontsize':font_size})
    plt.ylabel(ylabel, fontdict={'fontsize':font_size})
    plt.title(title, fontdict={'fontsize':font_size})
    plt.savefig(save_path, bbox_inches='tight')
    plt.cla()
    plt.close()


def plot_confocal_ECT(conf_image, ect_image, ect_cfg, confocal_cfg, save_path):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(conf_image, cmap='Reds')
    
    axarr[1].imshow(ect_image, 
                # vmin=np.mean(ect_image)+-4*np.std(ect_image), 
                # vmax=np.mean(ect_image)+1*np.std(ect_image), 
                cmap='Blues')  
    axarr[0].set_ylim(0, conf_image.shape[0])
    axarr[1].set_ylim(0, ect_image.shape[0])
    axarr[0].set_title('Confocal Image')
    axarr[1].set_title('ECT Image')
    axarr[0].set_xlabel('Columns')
    axarr[1].set_xlabel('Columns')
    axarr[0].set_ylabel('Rows')
    axarr[1].set_ylabel('Rows')
    
    # plot ECT boxes
    num_boxes = len(ect_cfg.COLUMNS)
    yrange = ect_cfg.COL_RANGE
    xrange = ect_cfg.ROWS 
    for i in range(num_boxes):
        anchor_point = (yrange[i][0], xrange[i][0])
        width, height = [yrange[i][1]-yrange[i][0]+1, xrange[i][1]-xrange[i][0]+1]

        rect = patches.Rectangle(anchor_point, width, height, linewidth=0.5, edgecolor='r', facecolor='none')
        axarr[1].add_patch(rect) 
        axarr[1].annotate(f"{i}", anchor_point, color='red', weight='normal', fontsize=6, ha='center', va='bottom')
        # axarr[1].plot([ect_cfg.COLUMNS[i], ect_cfg.COLUMNS[i]], [xrange[i][0], xrange[i][1]], linewidth=0.3, color="r")

    # plot Confocal boxes
    num_boxes = len(confocal_cfg.COLUMNS)
    yrange = confocal_cfg.COL_RANGE
    xrange = confocal_cfg.ROWS 
    for i in range(num_boxes):
        anchor_point = (yrange[i][0], xrange[i][0])
        width, height = [yrange[i][1]-yrange[i][0]+1, xrange[i][1]-xrange[i][0]+1]

        rect = patches.Rectangle(anchor_point, width, height, linewidth=0.5, edgecolor='r', facecolor='none')
        axarr[0].add_patch(rect) 
        axarr[0].annotate(f"{i}", anchor_point, color='black', weight='normal', fontsize=6, ha='center', va='bottom')
        # axarr[0].plot([confocal_cfg.COLUMNS[i], confocal_cfg.COLUMNS[i]], [xrange[i][0], xrange[i][1]], linewidth=0.3, color="r")

    plt.savefig(save_path)
    plt.close()

def draw_perm(mesh_points, mesh_triangles, mesh_params, perm, el_pos, num_electrodes, output_dir, colorbar=False, cmap=plt.cm.Greys):
    x, y = mesh_points[:, 0], mesh_points[:, 1]
   
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
  
    # draw mesh structure
    im = ax1.tripcolor(
        x,
        y,
        mesh_triangles,
        perm,
        edgecolors="k",
        shading="flat",
        cmap=cmap,
        vmin=0.6,
        vmax=0.9,
        alpha=0.2,
    ) 
    
    if colorbar:
        fig.colorbar(im, ax=ax1,orientation='horizontal')

    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], "ro")
    for i in range(num_electrodes):
        e = el_pos[i]
        ax1.text(x[e], y[e]-1e-6, str(i), size=12, horizontalalignment='center', verticalalignment='top')
    ax1.set_title("Permittivity Distribution")
    ax1.set_aspect("equal")
    ax1.set_ylim([mesh_params.offset, mesh_params.mesh_height])
    ax1.set_xlim([-mesh_params.mesh_width/2, mesh_params.mesh_width/2])
    
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax1.xaxis.set_major_formatter(ticks_x)        
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)      
    
    ax1.set_xlabel('microns')
    ax1.set_ylabel('microns (height)')

    # save image
    plt.savefig(output_dir)
    plt.close()


def draw_voltage(mesh_points, mesh_triangles, mesh_params, perm, ex_mat, voltage, el_pos, output_dir):
    x, y = mesh_points[:, 0], mesh_points[:, 1]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # draw equi-potential lines
    vf = np.linspace(min(voltage), max(voltage), 32)   # list of contour voltages
    ax1.tricontour(x, y, mesh_triangles, voltage, vf, cmap=plt.cm.inferno)
    
    # draw mesh structure
    ax1.tripcolor(
        x,
        y,
        mesh_triangles,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.2,
        cmap=plt.cm.Greys,
    )

    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], "ro")
    for i in range(mesh_params.number_electrodes):
        e = el_pos[i]
        ax1.text(x[e], y[e]-1e-6, str(i), size=12, horizontalalignment='center', verticalalignment='top')
    ax1.set_title(f"equi-potential lines for {ex_mat}")
  
    ax1.set_aspect("equal")
    ax1.set_ylim([mesh_params.offset, mesh_params.mesh_height])
    ax1.set_xlim([-mesh_params.mesh_width/2, mesh_params.mesh_width/2])
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax1.xaxis.set_major_formatter(ticks_x)        
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)      
    ax1.set_xlabel('microns')
    ax1.set_ylabel('microns (height)')

    plt.savefig(output_dir)
    plt.close()


def draw_electric_field(mesh_points, mesh_triangles, mesh_params, perm, electric_field, el_pos, output_dir):
    x, y = mesh_points[:, 0], mesh_points[:, 1]
    Ex, Ey = electric_field
    color = 2 * np.log(np.hypot(Ex, Ey))
    
    x_rgrid, y_rgrid = get_2d_grid()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.streamplot(x_rgrid,y_rgrid, Ex, Ey, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=1, arrowstyle='->', arrowsize=1.5)
    
    # draw mesh structure
    ax1.tripcolor(
        x,
        y,
        mesh_triangles,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.2,
        cmap=plt.cm.Greys,
    )

    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], "ro")
    for i in range(mesh_params.number_electrodes):
        e = el_pos[i]
        ax1.text(x[e], y[e]-1e-6, str(i + 1), size=12, horizontalalignment='center', verticalalignment='top')
   
    ax1.set_title("estimated electric field lines")
    # clean up
    ax1.set_aspect("equal")
    ax1.set_ylim([mesh_params.offset, mesh_params.mesh_height])
    ax1.set_xlim([-mesh_params.mesh_width/2, mesh_params.mesh_width/2])
    scale_x,scale_y = 1e-6,1e-6
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    ax1.xaxis.set_major_formatter(ticks_x)        
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)      
    ax1.set_xlabel('microns')
    ax1.set_ylabel('microns (height)')

    plt.savefig(output_dir)
    plt.close()


def draw_line(x, y, title, xlabel, ylabel, save_path):
    plt.plot(x, y, 'bs', linestyle="solid") 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


def draw_boundary_measurement(vb, save_path):
    combs = np.arange(MeshParams.number_electrodes-1, 0, -1)
    indx = 0
    for i, comb in enumerate(combs):
        start = indx
        end = start + comb  
        draw_line(vb[start:end], f"Boundary measurement {i}", save_path=os.path.join(save_path, f"vb_{i}.png"))
        indx += comb
    plt.close()


def draw_grid(grid, title, xlabel, ylabel, save_path, figsize=(6, 6), cmap='viridis', colorbar=True, scale_bar=False, scale_bar_length_fraction=0.35, scale_bar_box_alpha=0, scale_bar_text_color='w', ticks=True, font_size=10, format='png', aspect_ratio=1):
    """
        draw 2D grid
    """
    matplotlib.rcParams.update({'font.size': font_size})
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap=cmap)
    if colorbar:
        plt.colorbar()
    
    # add scale bar
    if scale_bar: 
        scalebar = ScaleBar(1, "um", length_fraction=scale_bar_length_fraction, width_fraction=0.04, color=scale_bar_text_color, location='upper right', label_loc='top', scale_loc='top', box_alpha=scale_bar_box_alpha)
        plt.gca().add_artist(scalebar)
    
    if not ticks: 
        plt.tick_params(left = False, right = False , labelleft=False,labelbottom = False, bottom = False)

    plt.gca().set_aspect(aspect_ratio)
    plt.ylim([0, grid.shape[0]])
    plt.xlabel(xlabel, fontdict={'fontsize':font_size})
    plt.ylabel(ylabel, fontdict={'fontsize':font_size})
    plt.title(title, fontdict={'fontsize':font_size})
    plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.cla()
    plt.close()  


def draw_confocal_grid(grid, title, xlabel, ylabel, save_path, figsize=(6, 6), cmap='viridis', colorbar=True, scale_bar=False, ticks=False, font_size=10, format='png', aspect_ratio=1):
    """
        draw 2D grid
    """
    matplotlib.rcParams.update({'font.size': font_size})
    plt.figure(figsize=figsize)

    plt.imshow(grid, cmap=cmap, vmin=-0.55, vmax=2.4)
    if colorbar:
        plt.colorbar()
    
    # add scale bar
    if scale_bar: 
        scalebar = ScaleBar(1, "um", length_fraction=0.35, width_fraction=0.04, color='0.2', location='upper right', label_loc='top', scale_loc='top', box_alpha=0)
        plt.gca().add_artist(scalebar)
    
    if not ticks: 
        plt.tick_params(left = False, right = False , labelleft=False,labelbottom = False, bottom = False)

    plt.gca().set_aspect(aspect_ratio)
    plt.ylim([0, grid.shape[0]])
    plt.xlabel(xlabel, fontdict={'fontsize':font_size})
    plt.ylabel(ylabel, fontdict={'fontsize':font_size})
    plt.title(title, fontdict={'fontsize':font_size})
    plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.cla()
    plt.close()  



def draw_image(image, title, xlabel, ylabel, save_path):
    plt.imshow(image, cmap='viridis')
    plt.ylim([0, image.shape[0]])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


def draw_ect(image, xrange, vrange, title, xlabel, ylabel, save_path):
    ymin,ymax=[0, image.shape[0]]
    xmin,xmax=xrange            
    vmin,vmax=vrange

    plt.imshow(image,
                vmin=np.mean(image[ymin:ymax,xmin:xmax])+vmin*np.std(image[ymin:ymax,xmin:xmax]), 
                vmax=np.mean(image[ymin:ymax,xmin:xmax])+vmax*np.std(image[ymin:ymax,xmin:xmax]), 
                cmap='Blues')
    plt.ylim([0, image.shape[0]])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def draw_greit(image, pts, tri, delta_perm, save_path):
    # show alpha
    fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))

    ax = axes[0]
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(delta_perm), shading="flat")
    ax.axis("equal")
    
    # ax.set_xlim([0, image.shape[0]])
    # ax.set_ylim([0, image.shape[1]])
    
    ax.set_title(r"$\Delta$ Conductivity")

    ax = axes[1]
    im = ax.imshow(np.real(image), interpolation="none", cmap=plt.cm.viridis)
    ax.axis("equal")
    
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([0, image.shape[0]])

    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.savefig(save_path)


def draw_bp(image, pts, tri, delta_perm, save_path):
    # draw
    fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
    # original
    ax = axes[0]
    ax.axis("equal")
    ax.set_title(r"Input $\Delta$ Conductivities")
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")

    # reconstructed
    ax1 = axes[1]
    im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, image, shading="flat")
    ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
    ax1.axis("equal")
    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.savefig(save_path)



def interpolate_perm(mesh_points, mesh_triangles, perm, mesh_params):
    tri_centers = np.mean(mesh_points[mesh_triangles], axis=1)    
    x_tri, y_tri = tri_centers[:, 0], tri_centers[:, 1]

    x_rgrid, y_rgrid = get_2d_grid(mesh_params)
    
    # create intezrpolator for permittivity
    # and get permittivity on a rectangular mesh
    triang = Triangulation(x_tri, y_tri)
    tci = LinearTriInterpolator(triang, perm)
    perm_xy = tci(x_rgrid,y_rgrid) 

    perm_xy[np.isnan(perm_xy)] = 1.0 

    dperm = tci.gradient(x_rgrid,y_rgrid)  # this outputs a derivative that is scale by 1e6!
    dperm[0][np.isnan(dperm[1])] = 0.0 
    dperm[1][np.isnan(dperm[1])] = 0.0 

    # convert dperm to micron grid, 
    dperm = (dperm[0]*1e-6, dperm[1]*1e-6)
    return perm_xy, dperm


def interpolate_voltage(mesh_points, mesh_triangles, voltage, mesh_params):
    x, y = mesh_points[:, 0], mesh_points[:, 1]

    x_rgrid, y_rgrid = get_2d_grid(mesh_params)    

    # get the potential inside the mesh 
    triang = Triangulation(x, y, triangles=mesh_triangles)
    tci = CubicTriInterpolator(triang, voltage)
    u_xy = tci(x_rgrid, y_rgrid)

    e_xy = tci.gradient(x_rgrid, y_rgrid) # this outputs a derivative that is scaled by 1e6!
    
    # convert dperm to micron grid, 
    e_xy = (e_xy[0]*1e-6, e_xy[1]*1e-6)

    return u_xy, e_xy


def sweep_frame(slice_col, confocal_column, row_range, confocal_row_range, ect_image, conf_image, minerva_data, simulated_data, cross_section, pyeit_mesh, cross_section_raw, save_path):
    fig = plt.figure(figsize=(8, 8))

    conf_ax = fig.add_subplot(2, 3, 1)
    conf_ax.set_title("Confocal Image")
    conf_ax.set_ylim([0, conf_image.shape[0]])
    conf_ax.plot([confocal_column, confocal_column], [confocal_row_range[0], confocal_row_range[1]], linewidth=1, color="r")
    conf_ax.imshow(conf_image)

    ect_ax = fig.add_subplot(2, 3, 2)
    ect_ax.set_title("ECT Image")
    ect_ax.set_ylim([0, ect_image.shape[0]])
    ect_ax.plot([slice_col, slice_col], [row_range[0], row_range[1]], linewidth=1, color="r")
    ect_ax.imshow(ect_image, cmap='Blues')

    conf_cs_ax = fig.add_subplot(3, 3, 3)
    conf_cs_ax.set_ylim([0, cross_section.shape[0]])
    conf_cs_ax.set_title(f"Cross Sectional Image {row_range}")

    conf_cs_ax.imshow(cross_section, 
            vmin=(np.mean(cross_section)-(2*np.std(cross_section))), 
            vmax=(np.mean(cross_section)+(12*np.std(cross_section))), 
            cmap='Reds')

    sim_ax = fig.add_subplot(3, 2, 5)
    sim_ax.set_title("Minerva Data Scaled")
    sim_ax.set_xlabel("Measurement Index")
    sim_ax.set_ylabel("Capacitenc (F)")
    sim_ax.set_ylim([0, cross_section.shape[0]])

    sim_ax.imshow(cross_section_raw, cmap='Reds')
    # sim_ax.plot(np.arange(0, len(simulated_data)), simulated_data)

    minerva_ax = fig.add_subplot(3, 2, 6)
    minerva_ax.set_xlabel("Measurement Index")
    minerva_ax.set_ylabel("Capacitenc (fF)")
    minerva_ax.set_title("Raw Minerva Data")

    minerva_ax.plot(np.arange(0, len(minerva_data)), minerva_data)

    fig.canvas.draw()
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    im  = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if save_path: 
        plt.savefig(save_path)

    plt.cla()
    plt.close()

    return im 



def sweep_frame_2(slice_col, confocal_column, row_range, confocal_row_range, ect_image, conf_image, minerva_data, simulated_data, cross_section, pyeit_mesh, save_path):
    fig = plt.figure(figsize=(8, 8))

    conf_ax = fig.add_subplot(1, 2, 1)
    conf_ax.set_title("Confocal Image")
    conf_ax.set_ylim([0, conf_image.shape[0]])
    conf_ax.plot([confocal_column, confocal_column], [confocal_row_range[0], confocal_row_range[1]], linewidth=1, color="r")
    conf_ax.imshow(conf_image)

    ect_ax = fig.add_subplot(1, 2, 2)
    ect_ax.set_title("ECT Image")
    ect_ax.set_ylim([0, ect_image.shape[0]])
    ect_ax.plot([slice_col, slice_col], [row_range[0], row_range[1]], linewidth=1, color="r")
    ect_ax.imshow(ect_image, cmap='Blues')

    
    plt.savefig(save_path)
    plt.cla()
    plt.close()



def plot_capacitence_3d(capacitence_3d, title, row_offsets, col_offsets, save_path):
    num_sets = capacitence_3d.shape[0]
    fig, axs = plt.subplots(num_sets, sharex=True, sharey=True)
    fig.suptitle(title)
    print(num_sets)
    for i in range(0, num_sets):
        ect_meas = np.array(capacitence_3d[i])
        axs[i].plot(np.arange(0, len(capacitence_3d[i])), ect_meas[:, 2])
        axs[i].set_xlabel("Measurement Index")
        axs[i].set_ylabel("Capacitence [fF]")
        # axs[i].set_title(f"Row Offset {row_offsets[i]}, Column Offset {col_offsets[i]}")
   
    plt.savefig(save_path)
    plt.cla()
    plt.close()
