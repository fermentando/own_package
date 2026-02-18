import numpy as np
import matplotlib.pyplot as plt
import os
from read_hdf5 import read_hdf5
from matplotlib.colors import LogNorm
from plot_2d_image import plot_projection
from stratified_box import StratifiedBox
import glob


# Define your 4 run paths
base_run_paths = [
    "/viper/ptmp/ferhi/StratDisk/noturb/r100/",
    "/viper/ptmp/ferhi/StratDisk/m0.1/r100_v2/",
    "/viper/ptmp/ferhi/StratDisk/m0.3/r100_v2/",
    "/viper/ptmp/ferhi/StratDisk/m0.5/r100_v2/burnin/"
]

#[
#    "/viper/ptmp/ferhi/StratDisk/noturb/r100/",
#    "/viper/ptmp/ferhi/StratDisk/m0.1/r100_v2/",
#    "/viper/ptmp/ferhi/StratDisk/m0.3/r100_v2/",
#    "/viper/ptmp/ferhi/StratDisk/m0.5/r100_v2/burnin/"
#]

# [
#    "/viper/ptmp/ferhi/StratDisk/chi1e3/noturb/r100_tcool/",
#    "/viper/ptmp/ferhi/StratDisk/chi1e3/m0.1/r100_tcool/",
#    "/viper/ptmp/ferhi/StratDisk/chi1e3/m0.3/r100_tcool/",
#    "/viper/ptmp/ferhi/StratDisk/chi1e3/m0.5/r100_tcool/"
#]

run_paths = []
tlim = -1
for path in base_run_paths:
    sim = StratifiedBox(path + "strat.in", path)
    dt = float(sim.reader.get('parthenon/output2', 'dt'))
    print(f"dt for {path}: {dt}")
    if 'noturb' in path:
        print(f"Calculating tlim for {path}...", len(glob.glob(path + "out/parthenon.prim.*.phdf")))
        tlim = len(glob.glob(path + "out/parthenon.prim.*.phdf")) * dt
        shift_time = 0.0
    else:
        shift_time = sim.t_inject


    # Import all snapshots for comparison
    snp_interval = shift_time + tlim / 1.2
    snp_index = int(snp_interval / dt)
    snp_path = os.path.dirname(path) + f"/out/parthenon.prim.{snp_index:05d}.phdf"
    run_paths.append(snp_path)
    



# Create figure with 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(16, 16))
# Define shared normalization
norm_plot = LogNorm(vmin=1e-24, vmax=1e-22)  # Adjust to your density range

# Storage for colorbar reference
im = None

# Loop through each path
for j, path in enumerate(run_paths):
    print(f"Processing snapshot: {path}")
    try:
        # Read data
        data = read_hdf5(path, fields=['T', 'rho'])
        rho = data['rho']
        
        # Use your original plotting function
        plot_dict = plot_projection(
            np.transpose(rho, (1, 0, 2)),
            view_dir=2,
            cmap='viridis',
            new_fig=False,
            cbar_flag=False,
            fig=fig,
            ax=axes[j],
            kwargs={'norm': norm_plot}
        )
        
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        
        # Add title with run info
        #r_val = extract_r(path)
        #mach_val = extract_mach(path)
        #axes[j].set_title(f'r={r_val:.1f}, M={mach_val:.1f}', 
        #                 fontsize=12, color='white')
        
        # Save the image for shared colorbar
        im = plot_dict['slc']
        
    except Exception as e:
        print(f"Error processing {path}: {e}")
        axes[j].axis('off')

# Final colorbar
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar_ax.tick_params(axis='x', which='both', color='white', labelcolor='black',
                    labelsize=16, length=6, direction='in')
cbar_ax.set_xlabel(r'$\rho \, [\mathrm{g \, cm}^{-3}]$', fontsize=20)

# Save and show
print('Saved to /u/ferhi/Figures/Comparative_analysis/comparative_density_projection_chi2.png')
plt.savefig('/u/ferhi/Figures/Comparative_analysis/comparative_density_projection_chi2.png', dpi=300, bbox_inches='tight')
plt.show()