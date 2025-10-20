import os
import numpy as np
import matplotlib.pyplot as plt
from generate_ics import load_params
import unyt
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors


def plot_vsfs_from_dirs(basedirs):
    plt.style.use('custom_plot')
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis for the plot

    # Define linestyles and rocket color palette
    linestyles = ['-', '-.', ':']  # Different line styles for different basedirs
    rocket_colors = sns.color_palette('rocket', n_colors=3)  # 3 colors for the 3 different snapshots

    # Create a list for color labels corresponding to the suffixes
    color_labels = ['010', '100', '150']  # Hardcoded since it's based on your suffixes

    for i, basedir in enumerate(basedirs):
        # Assign line style based on the basedir
        linestyle = linestyles[i % len(linestyles)]

        if '10kc' in os.path.basename(os.path.normpath(basedir)):
            suffixes = ['100', '150']  # If '10kc' in the base directory, use these suffixes
            label_dir = '10kc'  # Label for this basedir
        else:
            suffixes = ['003','010', '015']
            label_dir = ''
            for part in basedir.split(os.sep):  # Split basedir into parts
                if 'kc' in part:  # Check if 'kc' is in any part of the path
                    label_dir = part  # Set this part as the label
                    break  # Stop once we've found the part with 'kc'

            # Default to basedir if no "kc" is found
            if not label_dir:
                label_dir = os.path.basename(basedir)

        target_filenames = [f'3d_vsf_{s}.npz' for s in suffixes]  # Generate filenames for the target snapshots

        # Create a flag to add the label only once for each basedir
        label_added = False

        for j, fname in enumerate(target_filenames):
            sim_dir = os.path.join('/viper/ptmp/ferhi/d3rcrit/', basedir.split('d3rcrit/')[-1])

            params = load_params(os.path.join(sim_dir, 'ism.in'))  # Load parameters
            depth  = float(params['reader'].get('problem/wtopenrun', 'depth'))
            cloud_r  = float(params['reader'].get('problem/wtopenrun', 'r0_cgs'))
            stand_l = 0.1 * cloud_r * unyt.cm  # Convert to cm and then to parsecs
            stand_l = stand_l.to('pc')

            full_path = os.path.join(basedir, fname)  # Full path to the file

            try:
                data = np.load(full_path)  # Load the data from the file
                vsf = data['vsf']
                log_centers = data['log_centers']
                x_vals = 10**log_centers / stand_l  # Convert to the appropriate units

                mask = ~np.isnan(vsf)  # Remove NaN values
                x_vals_clean = x_vals[mask]
                vsf_clean = vsf[mask]

                # Add the label only once for the basedir
                label = f"{label_dir}"
                if not label_added:
                    label_added = True  # Mark the label as added

                # Add color for each snapshot using the index j
                color = rocket_colors[(j) % 3]  # Get color for the snapshot

                ax.plot(
                    x_vals_clean,
                    vsf_clean,
                    label=label if not label_added else "",  # Only label once
                    linestyle=linestyle,  # Use the determined linestyle for this basedir
                    color=color  # Use the appropriate color for each snapshot
                )

            except Exception as e:
                print(f"Error reading {full_path}: {e}")

    # Reference slope line ∝ l^{1/3}

    
    x_start = 0.1  # Starting x-value in data units
    y_start = 1  # Starting y-value in data units

    # Define slope (1/3) in log-log space
    x_end = 0.3
    slope = 1/3
    y_end = y_start * (x_end / x_start)**slope  # Apply slope scaling

    # Plot the slope reference line
    ax.plot([x_start, x_end], [y_start, y_end],
            color='black', linewidth=3, solid_capstyle='round')

    # Label the slope line
    ax.text(x_end * 0.65, y_end * 0.65,
            r'$\propto l^{1/3}_{3D}$', fontsize=14, ha='left', va='bottom')

    ax.set_xlabel(r'$l_{3D} = \left| r_i - r_j \right| [r_{cl,init}]$')
    ax.set_ylabel(r'$S_{1,(x)} = \langle | v_{(x),i} - v_{(x),j} | \rangle$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)

    # Add colorbar for the snapshots
    norm = mcolors.Normalize(vmin=0, vmax=2)  # Since there are 3 snapshots (0, 1, 2)
    sm = plt.cm.ScalarMappable(cmap='rocket', norm=norm)
    sm.set_array([])  # Set empty array for colorbar
    cbar = plt.colorbar(sm, ax=ax, ticks=[0, 1, 2])  # Pass the axis to the colorbar
    cbar.set_label('Snapshot')
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(color_labels)  # Set color labels to match the suffixes

    ax.legend()
    plt.tight_layout()
    plt.savefig('vsf_time_general.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    default_dir = '/u/ferhi/Figures/velocity_structure_function/d3rcrit/'
    runs = ['01kc/fv01', 'kc/fv01_v2']#, '10kc/fv01']
    dirs = [default_dir + run for run in runs]
    plot_vsfs_from_dirs(dirs)
