import os
import numpy as np
import glob
from scipy.ndimage import label
from utils import get_n_procs_and_user_args
import matplotlib.pyplot as plt
from read_hdf5 import read_hdf5  
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde



def compute_cluster_sizes(binary_field):
    labeled_array, num_features = label(binary_field)
    sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (label 0)
    return sizes


def process_density_and_plot_kde(filepath, outdir, n_jobs=4):
    data = read_hdf5(filepath, n_jobs=n_jobs)
    density = data['rho']
    print("The data has been read")

    binary_field = (density > 1e-25)
    cluster_sizes = compute_cluster_sizes(binary_field)
    if len(cluster_sizes) == 0:
        raise ValueError("No clusters found.")

    r_clusters = ( cluster_sizes) ** (1 / 3) / 8
    volumes = r_clusters[r_clusters > 0] # normalize

    if len(volumes) < 2:
        print(f"Not enough data to compute KDE for {filepath}")
        return

    # Kernel density estimation
    bins = np.logspace(np.log10(min(volumes)), np.log10(max(volumes)), 30)
    hist, bin_edges = np.histogram(volumes, bins=bins, density=True)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    plt.plot(bin_centers, hist, drawstyle='steps-mid', label='PDF of Clump Volumes')

    # Reference power law
    ref_x = np.logspace(np.log10(min(volumes)), np.log10(max(volumes)), 100)
    ref_y = 0.1 * ref_x**-4
    plt.plot(ref_x, ref_y, 'r--', label=r'$ \propto V^{-4}$')

    plt.fill_between(ref_x, ref_y.max()*10, ref_y.min()/10, where=ref_x < 1,
                     color='gray', alpha=0.3, interpolate=True)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$V \ [ r^3_\mathrm {cl} ]$')
    plt.ylabel(r'Density (arbitrary units)')
    plt.legend()

    idx = filepath.split('/')[-1].split('.')[-2]
    outfile = os.path.join(outdir, f"kde_dndm_{str(int(idx)).zfill(3)}")
    plt.savefig(outfile, dpi=300)
    plt.close()


def run_all_parallel(run_list, outdir, n_procs):
    os.makedirs(outdir, exist_ok=True)
    Parallel(n_jobs=max(1, n_procs // 4))(
        delayed(process_density_and_plot_kde)(run, outdir)
        for run in run_list
    )
    
    
if __name__ == "__main__":
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")

    if len(user_args) == 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
        print('Saved to: ', saveFile)
        output_dir = os.path.join('/u/ferhi/Figures/clump_distribution/', f"{parts[-3]}/{parts[-2]}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        runDir = os.getcwd()
        run_paths = np.array([
            os.path.join(runDir, folder)
            for folder in os.listdir(runDir)
            if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder))
        ])
        parts = runDir.split('/')
        saveFile = f"{parts[-2]}/{parts[-1]}"
        output_dir = os.path.join('/u/ferhi/Figures/clump_distribution/', parts[-2])
        os.makedirs(output_dir, exist_ok=True)

    for run_path in run_paths:
        single_file_paths = sorted(glob.glob(os.path.join(run_path, 'out/parthenon.prim.*.phdf')))
        if not single_file_paths:
            print(f"No PHDF files found in {run_path}, skipping...")
            continue

        sim_input = run_path.split('out')[0]
        param_file = os.path.join(sim_input, 'ism.in')
        if not os.path.isfile(param_file):
            print(f"Parameter file not found: {param_file}, skipping...")
            continue

    full_output_dir = os.path.join('/u/ferhi/Figures/clump_distribution/', saveFile)
    os.makedirs(full_output_dir, exist_ok=True)

    run_all_parallel(single_file_paths, outdir=full_output_dir, n_procs=N_procs)
