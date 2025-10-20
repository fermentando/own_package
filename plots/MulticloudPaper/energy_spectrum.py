import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import get_n_procs_and_user_args
from joblib import Parallel, delayed
from generate_ics import load_params
from adjust_ics import SingleCloudCC
from read_hdf5 import read_hdf5

def compute_energy_spectrum_cold(vx, vy, vz, T, box_length_prime = 1.0):
    cold_mask = T < 5e4  # Define a mask for cold gas, e.g., T < 100 K
    vx_cold = np.where(cold_mask, vx, 0)
    vy_cold = np.where(cold_mask, vy, 0)
    vz_cold = np.where(cold_mask, vz, 0)
    
    return compute_energy_spectrum(vx_cold, vy_cold, vz_cold, box_length_prime)

def compute_energy_spectrum(vx, vy, vz, T, box_length=1.0):
    fft_vx = np.fft.fftn(vx)
    #fft_vy = np.fft.fftn(vy)
    fft_vz = np.fft.fftn(vz)
    print(box_length)

    Nx, Ny, Nz = vx.shape
    kx = np.fft.fftfreq(Nx, d=box_length / Nx)
    ky = np.fft.fftfreq(Ny, d=box_length / Ny)
    kz = np.fft.fftfreq(Nz, d=box_length / Nz)

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    N = np.prod(vx.shape)  # Total number of points in the grid
    energy_k = 0.5 * (np.abs(fft_vx)**2 + np.abs(fft_vz)**2)/ N**2

    k_mag_flat = k_mag.flatten()
    energy_flat = energy_k.flatten()

    k_bins = np.arange(0.5, min(Nx, Ny, Nz)//2 + 1, 1)
    E_k = np.zeros(len(k_bins))

    for i, k in enumerate(k_bins):
        mask = (k_mag_flat >= k - 0.5) & (k_mag_flat < k + 0.5)
        E_k[i] = np.sum(energy_flat[mask])


    k_vals = k_bins / (2 * np.pi)  # Converts from angular frequency to inverse length
    return k_bins, E_k



def process_run(infile, stand_l, outdir, n_jobs):
    print(f"Processing {infile}")
    idx = infile.split('/')[-1].split('.')[-2]
    outfile = os.path.join(outdir, f"cold_energy_spectrum_{str(int(idx)).zfill(3)}.png")
    if os.path.exists(outfile):
        print(f"[✓] Skipping {outfile}, already exists.")
        return

    # Load simulation snapshot
    data = read_hdf5(infile, fields = ['vel1', 'vel2', 'vel3', 'T'], n_jobs=n_jobs)
    vx, vy, vz = data['vel1'], data['vel2'], data['vel3']
    T = data['T']

    # box length in physical units
    box_length = 1#vx.shape[1] / stand_l  

    k_vals, E_k_vals = compute_energy_spectrum(vx, vy, vz, T, box_length=box_length)

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.loglog(k_vals, E_k_vals, label=f'Energy Spectrum\n{idx}')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$E(k)$')
    plt.title('Velocity Power Spectrum')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

    print(f"[✓] Energy spectrum plot saved: {outfile}")


def run_all_parallel(run_list, stand_l, outdir, n_procs):
    os.makedirs(outdir, exist_ok=True)

    Parallel(n_jobs=max(1, n_procs // 4))(
        delayed(process_run)(infile, stand_l, outdir, n_jobs=4)
        for infile in run_list
    )


if __name__ == "__main__":
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")

    RUNS = [os.getcwd()]
    run_paths = RUNS
    parts = RUNS[0].split('/')
    saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
    outdir = os.path.join('/u/ferhi/Figures/energy_spectrum/', f"{parts[-3]}/{parts[-2]}/{parts[-1]}/")
    os.makedirs(outdir, exist_ok=True)
    print('Saving plots to:', outdir)

    single_file_paths = sorted(glob.glob(os.path.join(run_paths[0], 'out/parthenon.prim.[0-9]*.phdf')))[::-1]
    if not single_file_paths:
        raise RuntimeError(f"No output files found in {run_paths[0]}")

    sim_input = run_paths[0].split('out')[0]
    params = load_params(os.path.join(sim_input, 'ism.in'))
    depth = float(params['reader'].get('problem/wtopenrun', 'depth'))
    cloud_r = float(params['reader'].get('problem/wtopenrun', 'r0_cgs'))
    stand_l = 8


    run_all_parallel(single_file_paths, stand_l,  outdir, n_procs=N_procs)