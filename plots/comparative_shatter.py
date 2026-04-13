import numpy as np
import matplotlib.pyplot as plt
import os
from read_hdf5 import read_hdf5
from matplotlib.colors import LogNorm
from plot_2d_image import plot_projection
from stratified_box import StratifiedBox
import glob
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.style.use('custom_plot')
cmap = sns.color_palette("vlag", as_cmap=True)
cmap = sns.color_palette("icefire", as_cmap=True)

# Define your 4 run paths
base_run_paths = [
    '/viper/ptmp/ferhi/InfallTurbulent/mach_test/m0.1/500lshatter/',
    '/viper/ptmp/ferhi/InfallTurbulent/mach_test/m0.3/500lshatter/',
    '/viper/ptmp/ferhi/InfallTurbulent/mach_test/m0.5/500lshatter/'
]



#[
#    "/viper/ptmp/ferhi/StratDisk/noturb/r100/",
#    "/viper/ptmp/ferhi/StratDisk/m0.1/r100_v2/",
#    "/viper/ptmp/ferhi/StratDisk/m0.3/r100_v2/",
#    "/viper/ptmp/ferhi/StratDisk/m0.5/r100_v2/burnin/"
#]

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
    if 'm0.1' in path:
        print(f"Calculating tlim for {path}...", len(glob.glob(path + "out/parthenon.prim.*.phdf")))
        tlim = len(glob.glob(path + "out/parthenon.prim.*.phdf")) * dt 
        shift_time = sim.t_inject
    else:
        shift_time = sim.t_inject


    # Import all snapshots for comparison
    for i in range(4):
         snp_interval = shift_time  + i * tlim / 11
         snp_index = int(snp_interval / dt)
         if snp_index >= len(glob.glob(path + "out/parthenon.prim.*.phdf")) -1:
             snp_index = len(glob.glob(path + "out/parthenon.prim.*.phdf")) - 2
             print(f"Warning: Requested snapshot index {snp_index} exceeds available snapshots. Using index {snp_index} instead.")
         snp_path = os.path.dirname(path) + f"/out/parthenon.prim.{snp_index:05d}.phdf"
         run_paths.append(snp_path)
    



# Create figure with 1 row, 4 columns
fig = plt.figure(figsize=(8, 12))
gs = fig.add_gridspec(
    4, 4,   # 3 rows, 4 columns grid
    height_ratios=[1,1,1,0.03], 
    hspace = 0.1
)

# Your original plots would be in:
axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]
# Define shared normalization
norm_plot = LogNorm(vmin=1, vmax=100)  # Adjust to your density range

# Storage for colorbar reference
im = None

# Loop through each path
for j, path in enumerate(run_paths):
    print(f"Processing snapshot: {path}")
    try:
        # Read data
        data = read_hdf5(path, fields=['T', 'rho'])
        rho = data['rho'][:, np.shape(data['rho'])[1]//4:3*np.shape(data['rho'])[1]//4, :]  # Extract mid-plane slice
        rho=np.transpose(rho, (1, 0, 2)) / 1e-24  # Normalize by minimum density

        
        # Use your original plotting function
        plot_dict = plot_projection(
            rho,
            view_dir=2,
            cmap=cmap,
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


cbar_ax = fig.add_subplot(gs[3, :])

cbar = fig.colorbar(
    im,
    cax=cbar_ax,
    orientation='horizontal'
)

cbar.ax.tick_params(
    axis='x',
    which='both',
    color='white',
    labelcolor='black',
    labelsize=16,
    length=6,
    direction='in'
)

cbar.set_label(r'$\chi$', fontsize=20)
plt.tight_layout()

# Save and show
save_path = '/u/ferhi/Figures/Comparative_analysis/comparative_density_projection_chi2_500l.png'
print('Saved to ' + save_path)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()