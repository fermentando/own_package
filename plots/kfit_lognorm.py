import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import argparse
import matplotlib.pyplot as plt
import os
from utils import *
from adjust_ics import *
import glob



def compute_cluster_sizes(binary_field):
    """
    Identifies 3D clusters and computes their sizes.
    """
    labeled_field, num_clusters = ndimage.label(binary_field)  # 3D connected components
    print("Number of clusters:", num_clusters)
    cluster_sizes = np.bincount(labeled_field.ravel())[1:]  # Ignore background (0 label)
    return np.sort(cluster_sizes[cluster_sizes > 0])[::-1]  # Sorted descending

def lognorm_fit(input_dir, mu_values):


    sim = SingleCloudCC(os.path.join(input_dir, 'ism.in'), input_dir)
    params = sim.reader
    ICs, kval = sim._return_ICs()

    print("Loaded ICs shape:", ICs.shape)
    
    binary_field = np.where(ICs[:,:,:,0]< params['rho_cloud'], 0, 1)
    cluster_sizes = compute_cluster_sizes(binary_field)
    print("Total clusters found:", len(cluster_sizes))
    
    # Compute cluster volume scaling
    cell_vol = ((params['xmax1'] - params['xmin1']) / params['nx1'])**2 * \
               ((params['xmax2'] - params['xmin2']) / params['nx2'])
    r_clusters = (3 / (4 * np.pi) * cell_vol * cluster_sizes)**(1/3) / params['Rcloud']
    
    # Histogram binning
    num_bins = 50 if len(cluster_sizes) > 50 else max(10, len(cluster_sizes) // 2)
    
    # Fit lognormal distribution
    shape, loc, scale = stats.lognorm.fit(r_clusters)
    mu = np.log(scale)
    mu_values.append((kval, mu))
    print(f"Lognormal Fit: shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}, mu={mu:.4f}")
    
    return r_clusters, num_bins, shape, loc, scale


def main(input_dirs, output_file):
    mu_values = []

    num_dirs = len(input_dirs)

    # Determine subplot grid (e.g., 2 rows if more than 2 dirs)
    rows = int(np.ceil(np.sqrt(num_dirs)))
    cols = int(np.ceil(num_dirs / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten in case of non-square grid

    for i, input_dir in enumerate(input_dirs):
        if os.path.exists(os.path.join(input_dir, 'ICs.bin')):
            try:
                r_clusters, num_bins, shape, loc, scale = lognorm_fit(input_dir, mu_values)
            except:
                continue
            x = np.linspace(r_clusters.min(), r_clusters.max(), 100)

            ax = axes[i]
            print(ax)
            ax.hist(r_clusters, bins=num_bins, alpha=0.3, label='Histogram')
            #ax.plot(x, stats.lognorm.pdf(x, shape, loc, scale), '--', label='Lognorm Fit')

            ax.set_title(f"Dir: {input_dir}")
            ax.set_xlabel("Cluster Radius")
            ax.set_ylabel("Density")
            ax.legend()

    # Remove empty subplots if num_dirs < rows * cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('/u/ferhi/Figures/k_estimates.png')

    # Save mu values to file
    with open(output_file, 'w') as f:
        for kval, mu in mu_values:
            f.write(f"{kval} {mu}\n")


if __name__ == "__main__":
    ksdir = '/raven/ptmp/ferhi/ISM_slab/*kc/*'
    input_dirs = np.sort(glob.glob(ksdir))  # Update this with actual path
    main(input_dirs, 'kestimates.txt')
