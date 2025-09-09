import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from read_hdf5 import read_hdf5 
from utils import get_user_args, get_n_procs_and_user_args 
from adjust_ics import SingleCloudCC 
import h5py


def clump_cumulative_distribution(binary_field, size_threshold = 0):
    labeled_array, _ = label(binary_field)
    clump_sizes = np.bincount(labeled_array.ravel())[1:]  # exclude background
    r_clusters = (clump_sizes) ** (1 / 3) 
    clump_sizes_above = clump_sizes[r_clusters > size_threshold]
    return clump_sizes_above

N_procs, user_args = get_n_procs_and_user_args()
print(f"N_procs set to: {N_procs} processors.")


if len(user_args) > 0:
    RUNS = [os.getcwd()]
    run_paths = RUNS
    parts = RUNS[0].split('/')
    saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}/Analysis"
else:
    runDir = os.getcwd()
    run_paths = np.array([
        os.path.join(runDir, folder)
        for folder in os.listdir(runDir)
        if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder))
    ])
    parts = runDir.split('/')
    saveFile = f"{parts[-2]}/{parts[-1]}/Analysis"

save_path = os.path.join('/u/ferhi/Figures/clump_distribution/', saveFile)
os.makedirs(save_path, exist_ok=True)
print(f"RUNS: {run_paths}")


for j, run in enumerate(run_paths):
    print(run)

    sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
    code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
    files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
    rcrit = 8 * 10 * sim.tcoolmix/sim.tcc

    results_file = os.path.join(save_path, "num_clumps_over_time.npz")

    if os.path.exists(results_file):
        existing = np.load(results_file, allow_pickle=True)
        snapshot_indices = list(existing['snapshot_indices'])
        filenames = list(existing['filenames'])
        num_clumps = list(existing['num_clumps'])
        times = list(existing['times'])
        start_idx = len(snapshot_indices)
        print(f"Resuming from snapshot {start_idx}")
    else:
        snapshot_indices = []
        filenames = []
        num_clumps = []
        times = []
        start_idx = 0

    for idx, file in enumerate(files[start_idx:], start=start_idx):
        print(f"Processing file {file} ({idx})")
        data = read_hdf5(file, fields=['rho'], n_jobs=4)
        density = data['rho']
        binary_field = density > 1e-25  # threshold for clumps
        clump_sizes = clump_cumulative_distribution(binary_field, size_threshold=0)
        total_clumps = len(clump_sizes)

        # Use snapshot index or parse from filename if needed
        try:
            with h5py.File(os.path.join(run, 'out', f'parthenon.prim.{str(idx).zfill(5)}.phdf'), 'r') as f:
                time_in_cgs = f['Info'].attrs['Time'] * code_time_cgs
        except FileNotFoundError:
            with h5py.File(os.path.join(run, 'out', 'parthenon.prim.final.phdf'), 'r') as f:
                time_in_cgs = f['Info'].attrs['Time'] * code_time_cgs

        # Record data
        snapshot_indices.append(idx)
        filenames.append(os.path.basename(file))
        num_clumps.append(total_clumps)
        times.append(time_in_cgs)

        # Save plot
        plt.figure(figsize=(8, 5))
        plt.plot(times, num_clumps, marker='o', color='blue')
        plt.xlabel('Time [s]')
        plt.ylabel('Number of Clumps')
        plt.title('Clump Evolution Over Time')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(save_path, f'clumps.png')
        plt.savefig(plot_path)
        plt.close()

        # Save data
        np.savez_compressed(
            results_file,
            snapshot_indices=np.array(snapshot_indices),
            filenames=np.array(filenames),
            num_clumps=np.array(num_clumps),
            times=np.array(times)
        )
