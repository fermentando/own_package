"""Velocity related analysis stuff.
"""
import yt
import numpy as np
import h5py
from joblib import Parallel, delayed
from numba import njit
import os
import glob
from utils import get_n_procs_and_user_args
from generate_ics import load_params
import matplotlib.pyplot as plt
import unyt
import pandas as pd
from scipy.signal import medfilt
plt.style.use('custom_plot')
# ====================== PARAMS ===========================
cm_to_pc = 3.24078e-19

def load_covering_grid(infile):
    latest_file = infile
    ds = yt.load(latest_file, default_species_fields="ionized")
    ds.force_periodicity()
    # Load full domain
    cg_full = ds.covering_grid(level=ds.max_level,
                               left_edge=ds.domain_left_edge,
                               dims=ds.domain_dimensions)

    # Mask cold gas (temperature ≤ 3e4 K)
    T = cg_full['temperature']
    mask_cold = T <= 3e4

    # Get positions in pc
    x = cg_full['x'][mask_cold].to('pc')
    y = cg_full['y'][mask_cold].to('pc')
    z = cg_full['z'][mask_cold].to('pc')

    # Compute center of cold gas (in pc)
    x0 = np.median(x)
    y0 = np.median(y)
    z0 = np.median(z)
    center = [x0, y0, z0]

    # Compute radial distances and get r_max (95th percentile)
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    r_max = np.percentile(r, 95)

    # Define left edge and box size in code units
    center_code = ds.arr(center).to('code_length')
    r_max_code = ds.quan(r_max, 'pc').to('code_length')

    left_edge = center_code - r_max_code
    box_size = 2 * r_max_code

    # Determine number of cells
    dx = cg_full.dds[0]  # in code units
    N_cells = int(np.ceil(box_size / dx))
    block_size = [N_cells] * 3

    # Load new covering grid centered on cold gas
    cg = ds.covering_grid(level=ds.max_level,
                          left_edge=left_edge,
                          dims=block_size)

    return cg, dx.to('pc')


def extract_flat_data(cg, temp_cut=3e4):
    nHp = cg['number_density'][:]
    ne  = cg['El_number_density'][:]
    T   = cg['temperature'][:]
    vx  = cg['velocity_x'][:]/1e5
    vy  = cg['velocity_y'][:]/1e5
    vz  = cg['velocity_z'][:]/1e5
    x   = cg['x'][:]*cm_to_pc
    y   = cg['y'][:]*cm_to_pc
    z   = cg['z'][:]*cm_to_pc


    # Flatten and mask
    mask = T.flatten() <= temp_cut
    return {
        'vx': vx.flatten()[mask],
        'vy': vy.flatten()[mask],
        'vz': vz.flatten()[mask],
        'x': x.flatten()[mask],
        'y': y.flatten()[mask],
        'z': z.flatten()[mask],
        'Halpha': (ne.flatten() * nHp.flatten())[mask], 
    }

def project_velocity(data, bins):
    y = data['y'].value  # Convert from unyt to plain NumPy array
    z = data['z'].value  # Convert from unyt to plain NumPy array
    vx = data['vx'].value  # Convert from unyt to plain NumPy array
    Halpha = data['Halpha'].value  # Convert from unyt to plain NumPy array
    
    hist_vx, _, _ = np.histogram2d(y, z, bins=(bins, bins), weights=vx * Halpha)
    hist_Halpha, _, _ = np.histogram2d(y, z, bins=(bins, bins), weights=Halpha)
    projected_vx = np.divide(hist_vx, hist_Halpha, where=hist_Halpha > 0)
    projected_vx[hist_Halpha == 0] = np.nan
    return projected_vx

def generate_flat(vx_image, bins):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    y, z = np.meshgrid(bin_centers, bin_centers, indexing='ij')
    flat_y, flat_z, flat_vx = y.ravel(), z.ravel(), vx_image.ravel()
    mask = ~np.isnan(flat_vx)
    return flat_y[mask], flat_z[mask], flat_vx[mask]

def determine_n_ij(data, max_n_ij=None):
    num_points = len((data['x']))  

    n_ij = int(np.sqrt(num_points) / 10)
    if max_n_ij is not None:
        n_ij = min(n_ij, max_n_ij)  
    return n_ij

def determine_length(r_max, bin_resolution):

    length = int(np.log10(r_max) * bin_resolution)  # Bin resolution can be adjusted
    return length
if False:
    def generate_bins(dx, data, max_n_ij=None, bin_resolution=200):
        N_ij = determine_n_ij(data, max_n_ij=max_n_ij)
        
        r_max = np.max(np.sqrt(data['y']**2 + data['z']**2))
        length = determine_length(r_max, bin_resolution=bin_resolution)

        return np.linspace(0*r_max, r_max, length//2)
else:
    def generate_bins(dx, data, max_n_ij=None, bin_resolution=80, length=None):
        N_ij = determine_n_ij(data, max_n_ij=max_n_ij)
        
        # Automatically determine length based on r_max or use provided length
        r_max = np.max(np.sqrt(data['y']**2 + data['z']**2))  # Maximum separation distance
        length = determine_length(r_max, bin_resolution=bin_resolution)
        
        # Generate bins based on determined N_ij and length
        ij = [np.sqrt(i**2 + j**2 + k**2) for i in range(N_ij) for j in range(N_ij) for k in range(N_ij)]
        ij = np.unique(np.array(ij)[1:] * dx)[:5]
        log_centers = np.log10(ij)
        N_regular = length - len(log_centers)
        start = log_centers[-1] + (log_centers[-1] - log_centers[-2])
        log_full = np.concatenate((log_centers, np.linspace(start, np.log10(r_max), N_regular)))
        
        return log_full


def generate_bins_edges(bin_centers):
    bw = np.diff(bin_centers) / 2
    edges = np.empty(len(bin_centers) + 1)
    edges[1:-1] = bin_centers[:-1] + bw
    edges[0] = bin_centers[0] - bw[0]
    edges[-1] = bin_centers[-1] + bw[-1]
    return edges

@njit
def compute_vsf(y, z, v, bins_edges, vsf, counts):
    N = len(v)
    for m in range(N):
        for n in range(m+1, N):
            dv = np.abs(v[m] - v[n])
            dr = np.sqrt((y[m] - y[n])**2 + (z[m] - z[n])**2)
            bin_idx = np.searchsorted(bins_edges, dr) - 1
            if 0 <= bin_idx < len(vsf):
                vsf[bin_idx] += dv
                counts[bin_idx] += 1

def generate_vsf(data, dx, outname, stand_l, min_pairs = 2):
    y, z, v = data['y'], data['z'], data['vx']
    log_centers = generate_bins(dx, data)
    edges = generate_bins_edges(log_centers)
    bin_edges = 10**edges
    vsf = np.zeros(len(bin_edges) - 1)
    counts = np.zeros_like(vsf)
    compute_vsf(y, z, v, bin_edges, vsf, counts)
    vsf /= np.where(counts > 0, counts, 1)
    counts = np.zeros_like(vsf)
    
    compute_vsf(y, z, v, bin_edges, vsf, counts)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        vsf = np.where(counts > 0, vsf / counts, np.nan)

    # Mask low-count bins
    vsf = np.where(counts >= min_pairs, vsf, np.nan)
    plt.style.use('custom_plot')
    plt.figure(figsize=(8, 6))

    # Plot VSF against bin centers (log scale for bins)

     
    plt.plot(10**log_centers/stand_l, vsf, label='Velocity Structure Function', color='blue')

    # Set plot labels and title
    plt.xlabel('Separation Distance (pc)', fontsize=12)
    plt.ylabel('VSF', fontsize=12)

    # Add a grid and legend
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')

    # Display the plot
    plt.savefig(os.path.join(outname+'.png'), dpi=300)
    plt.clf()

    #with h5py.File(outname+'.h5', 'w') as f:
    #    f.create_dataset('vsf', data=vsf)
    #    f.create_dataset('bins', data=log_centers)

def process_run(infile, stand_l, outdir):
    print(infile)
    idx = infile.split('/')[-1].split('.')[-2]
    outfile = os.path.join(outdir, f"2d_vsf_{str(int(idx)).zfill(3)}")

    cg, dx = load_covering_grid(infile)
    data = extract_flat_data(cg)
    block_size = cg.ActiveDimensions.tolist()
    hw = (block_size[1] * cg.ds.index.grids[0].dds[1]).in_cgs().value * cm_to_pc / 2
    N_cells = block_size[1]
    bins = np.linspace(-hw, hw, N_cells + 1)
    proj_vx = project_velocity(data, bins)
    y, z, vx = generate_flat(proj_vx, bins)
    generate_vsf(data, dx, outfile, stand_l)
    print(f"[✓] VSF generated: {outfile}")
    #except Exception as e:
    #    print(f"[✗] Failed for {infile}: {e}")

def run_all_parallel(run_list, stand_l, outdir, n_procs):
    os.makedirs(outdir, exist_ok=True)
    Parallel(n_jobs=n_procs)(
        delayed(process_run)(run, stand_l, outdir)
        for run in run_list
    )


if __name__ == "__main__":
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    if len(user_args) == 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
        print('Saved to: ', saveFile)
        if not os.path.exists(os.path.join('/u/ferhi/Figures/velocity_structure_function/',f"{parts[-3]}/{parts[-2]}")): 
            os.makedirs(os.path.join('/u/ferhi/Figures/velocity_structure_function/',f"{parts[-3]}/{parts[-2]}"))


    else:
        runDir = os.getcwd()
        run_paths = np.array([
            os.path.join(runDir, folder) 
            for folder in os.listdir(runDir) 
            if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder)) 
        ])
        parts = runDir.split('/')
        saveFile = f"{parts[-2]}/{parts[-1]}"
        if not os.path.exists(os.path.join('/u/ferhi/Figures/velocity_structure_function/',parts[-2])): 
            os.makedirs(os.path.join('/u/ferhi/Figures/velocity_structure_function/',parts[-2]))
    
    single_file_paths = sorted(glob.glob(os.path.join(run_paths[0], 'out/parthenon.prim.*.phdf')))

    sim_input = run_paths[0].split('out')[0]
    params = load_params(os.path.join(sim_input, 'ism.in'))
    depth  = float(params['reader'].get('problem/wtopenrun', 'depth'))
    cloud_r  = float(params['reader'].get('problem/wtopenrun', 'r0_cgs'))
    stand_l = cloud_r * depth * unyt.cm * cm_to_pc
    run_all_parallel(single_file_paths, stand_l, outdir=os.path.join('/u/ferhi/Figures/velocity_structure_function/',saveFile), n_procs=N_procs)
