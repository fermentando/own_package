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


def process_density_and_plot(filepath, outdir, n_jobs=4):
    data = read_hdf5(filepath, n_jobs=n_jobs)
    density = data['rho'] 

    binary_field = (density > 1e-25)
    volumes, _ = clump_cumulative_distribution(binary_field)
    volumes = volumes[np.isfinite(volumes) & (volumes > 0)]/8**3

    sorted_volumes = np.sort(volumes)
    bins = np.logspace(np.log10(min(sorted_volumes)), np.log10(max(sorted_volumes)), num=30)
    ccdf_counts = [np.sum(sorted_volumes >= b) for b in bins]
    plt.step(bins, ccdf_counts, where='post')


    plt.plot(volumes, 10 * volumes.astype(float)**-0.8, 'r--', label=r'$ N \propto V^{-1}$')
    plt.fill_between(bins, 1e6, 1, where=bins < 1,
                    color='gray', alpha=0.3, interpolate=True)
    plt.xticks([1e-2, 1e0, 1e2, 1e4], [r'$10^{-2}$', r'$10^{0}$', r'$10^{2}$', r'$10^{4}$'])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$V \ [ r^3_\mathrm {cl} ]$')
    plt.ylim(bottom=1, top = max(ccdf_counts) * 1.5)
    plt.ylabel(r'$ N (\geq V)$')

    # Save plot        
    idx = filepath.split('/')[-1].split('.')[-2]
    outfile = os.path.join(outdir, f"dndm_{str(int(idx)).zfill(3)}")

    plt.savefig(outfile, dpi = 300)
    plt.close()



def run_all_parallel(run_list, outdir, n_procs):
    os.makedirs(outdir, exist_ok=True)
    Parallel(n_jobs=n_procs//4)(
        delayed(process_density_and_plot)(run, outdir)
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
