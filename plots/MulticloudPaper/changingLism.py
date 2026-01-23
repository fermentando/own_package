import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
from plot_2d_image import plot_projection
import read_hdf5 as rd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm, Normalize


def detect_cold_box(temp, threshold=5e4, padding=40 + 3*8):
    cold_mask = temp <= threshold
    coords = np.argwhere(cold_mask)
    if coords.size == 0:
        raise ValueError("No cold gas found.")
    zmin, ymin, xmin = coords.min(axis=0)
    return max(ymin - padding, 0)

colors = [
    (0.4, 0.6, 0.6),     # light desaturated teal
    (0.2, 0.4, 0.4),     # deep muted teal
    (0.0, 0.0, 0.0),     # black (center)
    (0.5, 0.2, 0.0),     # dark orange
    (1.0, 0.8, 0.2)      # bright yellow
]

#colors = [
#    (0.0, 0.0, 0.0),    # black
#    (0.0, 0.2, 0.1),    # dark green/teal
#    (0.2, 0.5, 0.1),    # greenish
#    (0.8, 0.7, 0.3),    # yellowish
#    (1.0, 0.95, 0.6),   # bright yellow/white
#]


cmap = LinearSegmentedColormap.from_list('purple_to_yellow', colors)

#Create colormap first
colors = ['#b58900', 'white', '#5e005e']   # Dark Yellow → White → Dark Purple
custom_cmap = LinearSegmentedColormap.from_list('yellow_white_purple', colors)

Hist = False
Proj = True
# Define parameters
baseDir = '/viper/ptmp/ferhi/LEGACY/fvLism/'
savename ='changingL_muti_volweighted'
vol = ['01kc/fv01_movie_2', '01kc/fv01_30r']#, 'kc/fv01_shorter']  # Only one row for now
snps = [5, 80, 170]



vmin, vmax = 1, 1e2  # Color scale normalization
im = None
subplot_width = 6.5   # width in inches per subplot
subplot_height = 2  # height in inches per subplot

fig_width = subplot_width * len(snps)
fig_height = subplot_height * len(vol) 
ref_shape = 0
plt.style.use('custom_plot')

if Proj: 
    height_ratios = [0.08] + [1.0] * len(vol)  # small first row + normal rows for volumes
    fig, axes = plt.subplots(nrows=len(vol) + 1, ncols=len(snps), figsize=(fig_width, subplot_height * (len(vol) + height_ratios[0])),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05, 'height_ratios': height_ratios})

    # turn off the small top-row axes (keeps space but empty)
    for ax in axes[0, :]:
        ax.axis('off')

    # use the remaining rows for the rest of the code (volumes)
    axes = axes[1:, :]

    # Ensure `axes` is always a 2D array (fixes single-row case)
    if len(vol) == 1:
        axes = np.expand_dims(axes, axis=0)

    norm_plot = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for i, v_i in zip([0,1],vol):
        #if i == 1:
        snps = [5, 60, 170]
        #if i == 2:
        #    snps = [1, 10, 30]
        #else:
        #    snps = [1,10,30]
        for j, snp in enumerate(snps):
            try:
                snapshot = glob.glob(os.path.join(baseDir+v_i+'/out', 'parthenon.prim.'+str(snp).zfill(5)+'.phdf'))[0]
            except:
                axes[i, j].axis('off')
                continue
            print('Processing snapshot: ', snapshot)
            
            #Load data and make projection
            read = rd.read_hdf5(snapshot, fields=['rho', 'T'], n_jobs = 4)
            rho = read['rho']/1e-26
            if i == 0:
                ref_shape = rho.shape[1]
            else:
                try:
                    
                    ymin = detect_cold_box(read['T'])
                    rho = rho[:, ymin:ymin + ref_shape, :]
                except ValueError as e:
                    print(f"Error in snapshot {snp} of volume {v_i}: {e}.\n Defaulting to box dims.")
                    rho = rho[:, 0:ref_shape, :]
                
            plt.style.use('custom_plot')
            
            # create two sub-axes that split the existing subplot into top and bottom halves
            bbox = axes[i, j].get_position()  # in figure coordinates
            # hide the original axes (we'll place two new axes on top)
            axes[i, j].set_visible(False)

            half_h = bbox.height / 2.0
            bottom_pos = [bbox.x0, bbox.y0, bbox.width, half_h]
            top_pos = [bbox.x0, bbox.y0 + half_h, bbox.width, half_h]

            bottom_ax = fig.add_axes(bottom_pos)
            top_ax = fig.add_axes(top_pos)

            # draw a dashed white dividing line at the shared boundary (in axes coordinates)
            # draw at y=1 for bottom_ax and y=0 for top_ax so it appears exactly on the seam
            bottom_ax.plot([0, 1], [1, 1], transform=bottom_ax.transAxes, color='white', linestyle='--', linewidth=1.2, zorder=20, clip_on=False)
            top_ax.plot([0, 1], [0, 0], transform=top_ax.transAxes, color='white', linestyle='--', linewidth=1.2, zorder=20, clip_on=False)

            # compute geometry & projection parameters
            view_dir = 2
            L = np.shape(rho)
            dim = len(L)

            x_dir = (view_dir + 1) % dim
            y_dir = (view_dir + 2) % dim
            z_dir = view_dir

            x_data = np.linspace(0, L[x_dir] / 240, num=L[x_dir] + 1)
            y_data = np.linspace(0, L[y_dir] / 240, num=L[y_dir] + 1)
            z_data = np.linspace(0, L[z_dir] / 240, num=L[z_dir] + 1)

            # Bottom: full projection occupying bottom half
            mid = L[x_dir] // 2
            slab_width = 8
            slab_end = min(mid + slab_width, L[x_dir])
            rho_top = rho.copy()
            bottom_slice = rho_top[0:mid, :, :]
            bottom_plot = plot_projection(bottom_slice, view_dir=view_dir, cmap=cmap,
                                          weight_data=None, new_fig=False, cbar_flag=False,
                                          fig=fig, ax=bottom_ax, kwargs={'norm': norm_plot})

            # Top: thin slab around the middle (occupies top half)
            mid = L[x_dir] // 2
            slab_width = 8
            slab_end = min(mid + slab_width, L[x_dir])
            top_rho = rho.copy()
            # take only the slab along the projection axis for the top plot
            # use slicing consistent with view_dir=2 -> slice axis=2
            top_slice = top_rho[:mid, :, mid:slab_end]
            top_plot = plot_projection(top_slice, view_dir=view_dir, cmap=cmap,
                                       weight_data=None, new_fig=False, cbar_flag=False,
                                       fig=fig, ax=top_ax, kwargs={'norm': norm_plot})

            # draw the same contour (from the full projection) on the bottom half
            weight_data = np.ones_like(rho)
            rho_proj = np.sum(rho * weight_data, axis=view_dir) / np.sum(weight_data, axis=view_dir)

            x_centers = 0.5 * (x_data[:-1] + x_data[1:])
            y_centers = 0.5 * (y_data[:-1] + y_data[1:])
            X, Y = np.meshgrid(y_centers, x_centers)

            contour_levels = [1e-25, 7e-25]
            bottom_ax.contour(
                X, Y,
                rho_proj,
                levels=contour_levels,
                colors='white',
                norm=LogNorm(),
                linewidths=0.7,
                alpha=0.4
            )

            # tidy axes: no ticks/labels on the small axes
            for a in (bottom_ax, top_ax):
                a.set_xticks([])
                a.set_yticks([])

            # keep a handle to the image for the colorbar (use bottom plot's image)
            if snp == snps[-1]:
                im = bottom_plot.get('slc', None)
   
   

                
        
    rs = [6,30]
    ts = [0, 0.5, 1]
    # Place readable row labels (left of each row) and column labels (above each column)
    for ii in range(len(vol)):
        # use the leftmost original-axis bbox to compute a nice text position
        bbox_row = axes[ii, 0].get_position()
        x_text = bbox_row.x0 - 0.01  # slightly more to the left for rotated text
        y_text = bbox_row.y0 + bbox_row.height / 2.0
        label = rf'$L_{{\mathrm{{ISM}}}} = {rs[ii]} r_{{\mathrm{{cl}}}}$'
        # make the row label vertical
        fig.text(x_text, y_text, label, fontsize=16, va='center', ha='center', rotation=90)

    for jj in range(len(snps)):
        bbox_col = axes[0, jj].get_position()
        x_text = bbox_col.x0 + bbox_col.width / 2.0
        y_text = bbox_col.y0 + bbox_col.height + 0.01  # slightly above the top
        label = rf'$t_\mathrm{{ent}} \sim {ts[jj]}$'
        fig.text(x_text, y_text, label, fontsize=16, va='bottom', ha='center')
    
    # Colorbar setup (apply to all subplots)
    plt.suptitle(r'$(r_\mathrm{cl} / r_\mathrm{crit}, f_\mathrm{v}) = (1, 10^{-1})$', x=0.51, y=0.97, fontsize=16)
    # lower `top` to push subplots down and create more space under the suptitle
    fig.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.15, top=0.88)  # more space between title and plots
    cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.06])  # [left, bottom, width, height] for horizontal bar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.tick_params(axis='x', which='both', color='white', direction='in', pad=10,    length=8,       # tick length
    width=2)
    cbar_ax.set_xlabel(r'$\chi$')
    print(f'Saving figure to /u/ferhi/Figures/{savename}.png')
    plt.savefig(f'/u/ferhi/Figures/{savename}.png',bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

