import os
import numpy as np
import glob
from scipy.ndimage import label
from utils import get_n_procs_and_user_args
import matplotlib.pyplot as plt
from read_hdf5 import read_hdf5  
from joblib import Parallel, delayed

def clump_cumulative_distribution(binary_field):
    """
    Given a binary field (e.g., final_percolation), return the cumulative number of clumps (N(>V))
    for clump sizes equal to or above the value.
    """
    labeled_array, _ = label(binary_field)
    clump_sizes = np.bincount(labeled_array.ravel())[1:]  # exclude background
    sorted_sizes = np.sort(clump_sizes)[::-1]
    cumulative_counts = np.arange(1, len(sorted_sizes) + 1)  # cumulative number of clumps
    return sorted_sizes, cumulative_counts


def process_density(filepath, n_jobs=4):
    data = read_hdf5(filepath, n_jobs=n_jobs)
    density = data['rho'] 
    print(f"Data read from {filepath}")

    binary_field = (density > 1e-25)
    volumes, _ = clump_cumulative_distribution(binary_field)
    volumes = volumes[np.isfinite(volumes) & (volumes > 0)] 
    
    #PDF
    r_clusters = ( volumes) ** (1 / 3) / 8
    bins = np.logspace(np.log10(min(r_clusters)), np.log10(max(r_clusters)), 30)
    hist, bin_edges = np.histogram(r_clusters, bins=bins, density=True)
    bins_r = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    #CDF

    sorted_volumes = np.sort(volumes)/ 8**3
    bins_v = np.logspace(np.log10(min(sorted_volumes)), np.log10(max(sorted_volumes)), num=30)
    ccdf_counts = [np.sum(sorted_volumes >= b) for b in bins_v]
    print(f"Max bin. = {max(bins_v)}")

    return bins_v, ccdf_counts, volumes, bins_r, hist

def plot_subplots(run_list, outdir):
    #assert len(run_list) == 4, "Exactly four runs are required."

    plt.style.use("custom_plot")
    fig, axs = plt.subplots(1, 1, figsize=(8,8))

    #Start plotting additional background curves
    print(f"Additional curves: {run_list[6:len(run_list)]}")
    for idx in range(6, len(run_list)):  # two subplots
            
        filepath = run_list[idx]
        print(f"Additional curve {idx}")

        plt.style.use("custom_plot")
        bins_v, ccdf_counts, volumes, bins_r, r_hist = process_density(filepath)

        for ax_id, ax in enumerate([axs[0,1], axs[1,1]]):
            print("this is ax_id", ax_id)
            if ax_id == 1:
                xvals = bins_v; yvals = ccdf_counts
                
            elif ax_id == 0:
                
                xvals = bins_r; yvals = r_hist; 
            ax.step(xvals, yvals, where='post', color = 'gray', alpha=0.3)

    #Plot main reference curves
    for i in range(2):  # two subplots
        for j in range(3):  # two runs per subplot
            
            idx = i * 3 + j
            try: filepath = run_list[idx]
            except: continue

            plt.style.use("custom_plot")
            bins_v, ccdf_counts, volumes, bins_r, r_hist = process_density(filepath)

            for ax_id in [0]:
                ax = axs
                xvals = bins_r; yvals = r_hist; 
                    
                        
                
                if 'fv02' in run_list[idx]: ax.step(xvals, yvals, where='post', color = "#4C72B0", linestyle=':')
                else: ax.step(xvals, yvals, where='post')
                
                
                ax.fill_between(xvals, 1e6, 1e-4, where=xvals < 1, color='whitesmoke', interpolate=True)
                ax.set_xscale('log')
                ax.set_yscale('log')

                    
                
                ax.set_xlabel(r'$r \ [ r_\mathrm {cl} ]$')
                x0, x1 = 2, 3 
                y0 = 4           

                # Draw the line: y = y0 * (x/x0)^-1
                x_vals = np.array([x0, x1])
                y_vals = y0 * (x_vals / x0)**-2
                
                x_m2 = np.array([x0, x1])
                y_m2 = y0 * (x_m2/x0)**-4

                #ax.plot(x_vals, y_vals, color='k', linewidth=1)
                #ax.plot(x_m2, y_m2, color='k', linewidth=1)

                # Add annotation
                #ax.text(x1 * 1.2, y_vals[1] * 1.2, r'$r^{-2}$', color='k', fontsize = 16, verticalalignment='top')
                #ax.text(x_m2[1] * 1.1,  y_m2[1] * 0.9, r'$r^{-4}$', color='k', fontsize = 16, verticalalignment='top')
                # Draw a band around the line y = mx + c passing through (1,1) and (10, 1e-3)
                x_band = np.logspace(0, 1000, 1000)  # from 1 to 10
                # Calculate slope and intercept in log-log space
                x1, y1 = 1, 12
                m = -4
                c = np.log10(y1) - m * np.log10(x1)
                y_band = 10**(m * np.log10(x_band) + c)
                # Band: ±0.3 dex in y
                y_upper = y_band * 10**0.3
                y_lower = y_band * 10**-0.3
                ax.fill_between(x_band, y_lower, y_upper, color='orange', alpha=0.2, label=r'$r^{-4} \pm 0.3 \ \mathrm{dex}$')
                ax.set_ylim(1e-3,1e1)
                ax.set_xlim(0.8,1e2)

                


    axs.set_ylabel(r'$ N (\geq V)$')

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    axs.legend(loc='upper right')
    print(f'Saving figure to {os.path.join(outdir, "scalefree.pdf")}')
    plt.savefig(os.path.join(outdir, "scalefree.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

# Example usage


if __name__ == "__main__":

    runs = ['/viper/ptmp/ferhi/LEGACY/fvLism/01kc/scalefree_long/out/parthenon.prim.00000.phdf']
    plot_subplots(runs, '.')#'/u/ferhi/Figures/')
# plot_subplots(["run1.h5", "run2.h5", "run3.h5", "run4.h5"], "output_directory")
"""
    runs = [
        '/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00000.phdf',
        '/viper/ptmp/ferhi/fvLism/01kc/fv02/out/parthenon.prim.00000.phdf',
        '/viper/ptmp/ferhi/fvLism/kc/fv01_shorter/out/parthenon.prim.00000.phdf',
        '/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00090.phdf',
        '/viper/ptmp/ferhi/fvLism/01kc/fv02/out/parthenon.prim.00024.phdf',
        '/viper/ptmp/ferhi/fvLism/kc/fv01_shorter/out/parthenon.prim.00020.phdf', 

        '/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00172.phdf', 
        '/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00070.phdf', 
        '/viper/ptmp/ferhi/fvLism/01kc/fv02/out/parthenon.prim.00023.phdf',
        '/viper/ptmp/ferhi/fvLism/01kc/fv02/out/parthenon.prim.00021.phdf',
        #'/viper/ptmp/ferhi/fvLism/kc/fv01_shorter/out/parthenon.prim.00015.phdf', 
        '/viper/ptmp/ferhi/fvLism/kc/fv01_shorter/out/parthenon.prim.00027.phdf', 

    ]
"""