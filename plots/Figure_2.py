import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
from plot_2d_image import plot_projection
import read_hdf5 as rd
from matplotlib.colors import LinearSegmentedColormap


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

cmap = LinearSegmentedColormap.from_list('purple_to_yellow', colors)

#Create colormap first
colors = ['#b58900', 'white', '#5e005e']   # Dark Yellow → White → Dark Purple
custom_cmap = LinearSegmentedColormap.from_list('yellow_white_purple', colors)

Hist = False
Proj = True
# Define parameters
baseDir = '/viper/ptmp/ferhi/fvLism/'
savename ='simple_multiplot'
vol = ['01kc/fv01_30r', '01kc/fv01_movie_2', 'kc/fv01_shorter']  # Only one row for now
snps = [5, 80, 170]



vmin, vmax = 1e-26, 1e-24  # Color scale normalization
im = None
subplot_width = 6.5   # width in inches per subplot
subplot_height = 2  # height in inches per subplot

fig_width = subplot_width * len(snps)
fig_height = subplot_height * len(vol) 
ref_shape = 0
plt.style.use('custom_plot')

if Proj: 
    fig, axes = plt.subplots(nrows=len(vol), ncols=len(snps), figsize=(fig_width, fig_height), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # Ensure `axes` is always a 2D array (fixes single-row case)
    if len(vol) == 1:
        axes = np.expand_dims(axes, axis=0)

    norm_plot = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for i, v_i in zip([1,2,0],vol):
        if i == 0:
            snps = [1,14,29]
        for j, snp in enumerate(snps):
            try:
                snapshot = glob.glob(os.path.join(baseDir+v_i+'/out', 'parthenon.prim.'+str(snp).zfill(5)+'.phdf'))[0]
            except:
                axes[i, j].axis('off')
                continue
            print('Processing snapshot: ', snapshot)
            
            #Load data and make projection
            read = rd.read_hdf5(snapshot, fields=['rho', 'T'], n_jobs = 4)
            rho = read['rho']
            if i == 1: ref_shape = rho.shape[1]
            if i == 0:
                ymin = detect_cold_box(read['T'])
                rho = rho[:, ymin:ymin + ref_shape, :]
                
            plt.style.use('custom_plot')
            
            plot_dict = plot_projection(rho, view_dir=2, cmap=cmap, new_fig=False, cbar_flag = False, fig = fig, ax=axes[i, j], kwargs={'norm': norm_plot})

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            

            if snp == snps[-1]:
                im = plot_dict['slc']
   
   

                
        
    ks = [10,1,1]
    ts = [0,2,4]
    for i in range(3):
        plt.style.use('custom_plot')
        axes[i, 0].set_ylabel(rf'$k/k_{{\mathrm{{crit}}}} = {ks[i]}$', fontsize = 16, labelpad = 8)
        axes[i, 0].yaxis.set_label_position("left")
        
        
        axes[0, i].xaxis.set_label_position('top') 
        axes[0, i].set_xlabel(rf'$t \sim {ts[i]} \, \tilde{{t}}_{{\mathrm{{cc}}}}$', fontsize = 16, labelpad = 8)
            
    

    # Colorbar setup (apply to all subplots)
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1, right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.01, 0.7])  # Position of the colorbar
    fig.colorbar(im, cax=cbar_ax)
    #for spine in cbar_ax.spines.values():
    #    spine.set_visible(False)
    cbar_ax.tick_params(axis='y', which='both', color = 'white', direction='in')
    cbar_ax.set_ylabel(r'$\rho \, [\mathrm{cm}^{-3}]$')
    plt.savefig(f'/u/ferhi/Figures/{savename}.pdf',bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()


# -------- Hist -----------#
if Hist:
    fig, axes = plt.subplots(nrows=len(vol), ncols=len(snps), figsize=(15, 9), gridspec_kw={'wspace': 0.03, 'hspace': 0.03})
    # Ensure `axes` is always a 2D array (fixes single-row case)
    if len(vol) == 1:
        axes = np.expand_dims(axes, axis=0)
# Ensure `axes` is always a 2D array for consistency
    for i, v_i in enumerate(vol):
        for j, snp in enumerate(snps):
            try:
                snapshot = glob.glob(os.path.join(baseDir + v_i + '/out', 'parthenon.prim.' + str(snp).zfill(5) + '.phdf'))[0]
            except (IndexError, FileNotFoundError):
                # If file not found, make blank black subplot
                axes[i, j].axis('off')
                axes[i, j].set_facecolor('black')
                continue
            
            print('Processing snapshot:', snapshot)
            
            # Load data
            rho = rd.read_hdf5(snapshot, fields=['rho'])['rho'].flatten()

            # Filter out non-physical values (optional)
            rho = rho[rho > 0]

            # Plot histogram
            # Define logarithmic bins
            log_bins = np.logspace(np.log10(vmin), np.log10(vmax), 50)
            axes[i, j].hist(rho, bins=log_bins, color='purple', alpha=0.7, log=True,  histtype='stepfilled')
            axes[i, j].set_xlim(vmin, vmax)
            axes[i, j].set_xscale('log')
            axes[i, j].set_yscale('log')
            
            # Remove ticks for a cleaner look
            #axes[i, j].set_xticks([])
            #axes[i, j].set_yticks([])

    # Final adjustments
    plt.tight_layout()
    plt.show()