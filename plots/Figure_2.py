import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
from plot_2d_image import plot_projection
import read_hdf5 as rd

Hist = False
Proj = True
# Define parameters
baseDir = '/viper/ptmp/ferhi/fvLism/'
savename ='simple_multiplot'
vol = ['01kc/fv01_30r', '01kc/fv01_movie_2', '02kc/fv01']  # Only one row for now
snps = [5, 47, 74]



vmin, vmax = 1e-26, 1e-24  # Color scale normalization
im = None
subplot_width = 6.5   # width in inches per subplot
subplot_height = 2  # height in inches per subplot

fig_width = subplot_width * len(snps)
fig_height = subplot_height * len(vol)


if Proj: 
    fig, axes = plt.subplots(nrows=len(vol), ncols=len(snps), figsize=(fig_width, fig_height), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # Ensure `axes` is always a 2D array (fixes single-row case)
    if len(vol) == 1:
        axes = np.expand_dims(axes, axis=0)

    norm_plot = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for i, v_i in enumerate(vol):
        for j, snp in enumerate(snps):
            try:
                snapshot = glob.glob(os.path.join(baseDir+v_i+'/out', 'parthenon.prim.'+str(snp).zfill(5)+'.phdf'))[0]
                print(snapshot)
            except:
                axes[i, j].axis('off')
                continue
            print('Processing snapshot: ', snapshot)
            
            #Load data and make projection
            rho = rd.read_hdf5(snapshot, fields=['rho'])['rho']
            plt.style.use('custom_plot')
            
            plot_dict = plot_projection(rho, view_dir=2, cmap='magma_r', new_fig=False, cbar_flag = False, fig = fig, ax=axes[i, j], kwargs={'norm': norm_plot})

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            if snp == snps[-1]:
                im = plot_dict['slc']
            
    

    # Colorbar setup (apply to all subplots)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.01, 0.7])  # Position of the colorbar
    fig.colorbar(im, cax=cbar_ax)
    for spine in cbar_ax.spines.values():
        spine.set_visible(False)
    cbar_ax.tick_params(axis='y', which='both', color='white', direction='in')
    cbar_ax.set_ylabel(r'$\rho \, [\mathrm{cm}^{-3}]$')
    plt.savefig(f'/u/ferhi/Figures/{savename}.png',bbox_inches='tight')
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