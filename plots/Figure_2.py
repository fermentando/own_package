import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
from plot_2d_image import plot_projection
import read_hdf5 as rd

Hist = True
Proj = False
# Define parameters
baseDir = '/raven/ptmp/ferhi/RavenGPU/ISM_slab/'
vol = ['fv01e','']  # Only one row for now
snps = [1 , 7, 12]
vmin, vmax = 1e-26, 1e-24  # Color scale normalization
if Proj: 
    fig, axes = plt.subplots(nrows=len(vol), ncols=len(snps), figsize=(15, 9), gridspec_kw={'wspace': 0.03, 'hspace': 0.03})

    # Ensure `axes` is always a 2D array (fixes single-row case)
    if len(vol) == 1:
        axes = np.expand_dims(axes, axis=0)

    norm_plot = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for i, v_i in enumerate(vol):
        for j, snp in enumerate(snps):
            try:
                snapshot = glob.glob(os.path.join(baseDir+v_i+'/out', 'parthenon.prim.'+str(snp).zfill(5)+'.phdf'))[0]
            except:
                axes[i, j].axis('off')
                continue
            print('Processing snapshot: ', snapshot)
            
            #Load data and make projection
            rho = rd.read_hdf5(snapshot, fields=['rho'])['rho']
            
            plot_dict = plot_projection(rho, view_dir=2, cmap='magma_r', new_fig=False, cbar_flag = False, fig = fig, ax=axes[i, j], kwargs={'norm': norm_plot})

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            im = plot_dict['slc']
            
    

    # Colorbar setup (apply to all subplots)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])  # Position of the colorbar
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(axis='y', which='both', color='white', labelcolor='black', labelsize=16, length = 6, direction='in')
    cbar_ax.set_ylabel(r'$\rho \, [\mathrm{cm}^{-3}]$', fontsize=20)
    plt.savefig('/u/ferhi/Figures/thinslab_Figure_2.png')
    plt.show()
    plt.clf()


# -------- Hist -----------#
if Hist:
    fig, axes = plt.subplots(nrows=len(vol), ncols=len(snps), figsize=(15, 9), gridspec_kw={'wspace': 0.03, 'hspace': 0.03})
    nbins = 50
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
            axes[i, j].hist(rho, bins=nbins, color='purple', alpha=0.7, log=True,  histtype='stepfilled')
            axes[i, j].set_xlim(vmin, vmax)
            axes[i, j].set_xscale('log')
            axes[i, j].set_yscale('log')
            
            # Remove ticks for a cleaner look
            #axes[i, j].set_xticks([])
            #axes[i, j].set_yticks([])

    # Final adjustments
    plt.tight_layout()
    plt.show()