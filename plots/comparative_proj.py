import numpy as np
import matplotlib.pyplot as plt
import os
from read_hdf5 import read_hdf5
from matplotlib.colors import LogNorm
from plot_2d_image import plot_projection

plt.style.use("custom_plot")
def detect_cold_box(temp, threshold=1e4, padding=5):
    cold_mask = temp < threshold
    coords = np.argwhere(cold_mask)
    if coords.size == 0:
        raise ValueError("No cold gas found.")
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    shape = temp.shape
    return (
        (max(zmin - padding, 0), min(zmax + padding, shape[0])),
        (max(ymin - padding, 0), min(ymax + padding, shape[1])),
        (max(xmin - padding, 0), min(xmax + padding, shape[2]))
    )

# File paths
file_paths = [
    '/viper/ptmp2/ferhi/d3rcrit/10kc/fv01/out/parthenon.prim.00000.phdf',
    #'/viper/ptmp2/ferhi/d3rcrit/kc/fv01_v2/out/parthenon.prim.00000.phdf',
    #'/viper/ptmp2/ferhi/d3rcrit/01kc/fv01/out/parthenon.prim.00000.phdf'
]

# Detect cold region from first snapshot
print("Detecting cold gas region...")
data0 = read_hdf5(file_paths[0], fields=['T', 'rho'])
zrange, yrange, xrange_ = detect_cold_box(data0['T'], threshold=3e4, padding=5)
ref_shape = None

# Prepare figure
dy = yrange[1] - yrange[0]
dx = xrange_[1] - xrange_[0]

aspect_ratio = dy / dx
max_width = 12  # total inches across all subplots
width_per_subplot = max_width / len(file_paths)
height = width_per_subplot * aspect_ratio
fig, axes = plt.subplots(
    1, len(file_paths),
    figsize=(max_width, height),
    squeeze=False,
    gridspec_kw={'wspace': 0.2}
)
norm_plot = LogNorm(vmin=1e-26, vmax=1e-24)  # Adjust as needed
im = None

# Loop through each path
for j, path in enumerate(file_paths):
    print(f"Processing snapshot: {path}")
    try:
        if path == file_paths[0]:
            rho = data0['rho'][xrange_[0]:xrange_[1], yrange[0]:yrange[1], zrange[0]:zrange[1]]
        else:
            data = read_hdf5(path, fields=['T', 'rho'])
            zind, yind, xind = detect_cold_box(data['T'], threshold=3e4, padding=5)
            rho = data['rho'][xind[0]:xind[1], yind[0]:yind[1], zind[0]:zind[1]]
            
        if ref_shape is None:
            ref_shape = rho.shape
        else:
            pad_y = ref_shape[1] - rho.shape[1]
            if pad_y > 0:
                # Pad along the y-axis (axis=1)
                rho = np.pad(rho, ((0, 0), (0, pad_y), (0, 0)), mode='constant', constant_values=1e-26)

        # Use your original plotting function
        plot_dict = plot_projection(
            np.transpose(rho, (1, 0, 2)),
            view_dir=2,
            cmap='magma_r',
            new_fig=False,
            cbar_flag=False,
            fig=fig,
            ax=axes[0, j],
            kwargs={'norm': norm_plot}
        )

        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])

        # Save the image for shared colorbar
        im = plot_dict['slc']

    except Exception as e:
        print(f"Error processing {path}: {e}")
        axes[0, j].axis('off')

# Final colorbar
fig.subplots_adjust()
cbar_ax = fig.add_axes([0.1, 0.2, 0.7, 0.005])  
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar_ax.tick_params(axis='x', which='both', color='white', labelcolor='black',
                    labelsize=16, length=6, direction='in')
cbar_ax.set_xlabel(r'$\rho \, [\mathrm{cm}^{-3}]$', fontsize=20)

# Save and show
plt.savefig('/u/ferhi/Figures/comparative_cold_projection.png', dpi=300)
plt.show()
plt.clf()
