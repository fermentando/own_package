import numpy as np
import h5py
from joblib import Parallel, delayed
from numba import njit
import os
import glob
from utils import get_n_procs_and_user_args
from generate_ics import load_params
from adjust_ics import SingleCloudCC
from read_hdf5 import *
import unyt
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
plt.style.use('custom_plot')
# ====================== PARAMS ===========================
cm_to_pc = 3.24078e-19

import numpy as np
def load_cold_gas_region_and_fields(filename, mbar, temp_cut=3e4, n_jobs = 1):
    fields =  ['rho', 'prs', 'vel1', 'vel2', 'vel3', 'x1', 'x2', 'x3', 'T']
    cg = read_hdf5(filename, fields, n_jobs)

    # Extract raw fields
    T   = cg['T']           # K
    vx  = cg['vel1'] / 1e5  # km/s
    vy  = cg['vel2'] / 1e5
    vz  = cg['vel3'] / 1e5
    x   = cg['x1'] * cm_to_pc  # pc
    y   = cg['x2'] * cm_to_pc
    z   = cg['x3'] * cm_to_pc

    n_e = cg['rho'] / mbar
    n_p = n_e

    # Mask cold gas
    cold_mask = T <= temp_cut

    # Cold gas positions
    x_cold = x[cold_mask]
    y_cold = y[cold_mask]
    z_cold = z[cold_mask]

    # Compute cold gas center
    x0 = np.median(x_cold)
    y0 = np.median(y_cold)
    z0 = np.median(z_cold)

    # Compute radial distance from center for cold gas
    r = np.sqrt((x_cold - x0)**2 + (y_cold - y0)**2 + (z_cold - z0)**2)
    r_max = np.percentile(r, 95)

    # Clip entire dataset to box around cold gas
    dx = np.mean(np.diff(np.unique(x)))  # Assumes regular grid
    half_box = r_max
    

    region_mask = (
        (np.abs(x - x0) <= half_box) &
        (np.abs(y - y0) <= half_box) &
        (np.abs(z - z0) <= half_box)
    )

    final_mask = cold_mask & region_mask

    final_fields = {
        'vx': vx[cold_mask],
        'vy': vy[cold_mask],
        'vz': vz[cold_mask],
        'x': x[cold_mask],
        'y': y[cold_mask],
        'z': z[cold_mask],
        'Halpha': (n_e * n_p)[cold_mask],
    }
    
    print("file succesfully read!")
    return final_fields, dx


def project_velocity(data, bins):
    """Project the velocity along the 3D grid."""
    # Extract 3D data for x, y, z positions and velocities (vx, vy, vz)
    x = data['x']
    y = data['y']
    z = data['z']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    Halpha = data['Halpha']
    

    # Create 3D histograms for velocity components and Halpha weighting
    hist_vx, _, _, _ = np.histogram2d(x, y, bins=(bins, bins), weights=vx)
    #hist_vy, _, _, _ = np.histogram2d(x, z, bins=(bins, bins), weights=vy)
    hist_vz, _, _, _ = np.histogram2d(y, z, bins=(bins, bins), weights=vz)
    hist_Halpha, _, _, _ = np.histogram2d(x, y, bins=(bins, bins), weights=Halpha)
    
    # Calculate the projected velocities for vx, vy, vz
    projected_vx = np.divide(hist_vx, hist_Halpha, where=hist_Halpha > 0)
    #projected_vy = np.divide(hist_vy, hist_Halpha, where=hist_Halpha > 0)
    projected_vz = np.divide(hist_vz, hist_Halpha, where=hist_Halpha > 0)

    # Handle cases where there is no data
    projected_vx[hist_Halpha == 0] = np.nan
    #projected_vy[hist_Halpha == 0] = np.nan
    projected_vz[hist_Halpha == 0] = np.nan
    
    # Combine the velocities into one 3D array (stack the components)
    projected_velocity = np.stack((projected_vx, projected_vz), axis=-1)
    
    return projected_velocity



def generate_flat(vx_image, bins):
    """Generate a flattened 3D grid for x, y, z, and vx."""
    # Generate bins centered
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Create a 3D meshgrid for x, y, and z (since it's 3D)
    flat_x, flat_y, flat_z = np.meshgrid(bin_centers, bin_centers, bin_centers, indexing='ij')
    
    # Flatten the arrays into 1D vectors
    flat_x = flat_x.ravel()
    #flat_y = flat_y.ravel()
    flat_z = flat_z.ravel()
    
    # Ensure the number of elements in vx_image matches the flattened grid size
    flat_vx = vx_image.ravel()
    if len(flat_vx) != len(flat_x):
        raise ValueError(f"Shape mismatch: {len(flat_vx)} (vx_image) vs {len(flat_x)} (flattened grid)")
    
    # Filter out NaN values from the vx_image (velocity) array
    mask = ~np.isnan(flat_vx)
    
    # Apply the mask to get valid values
    x_values = flat_x[mask]
    #y_values = flat_y[mask]
    z_values = flat_z[mask]
    vx_values = flat_vx[mask]
    
    return x_values, z_values, vx_values


def determine_n_ij(data, max_n_ij=None):
    num_points = len((data['x']))  

    n_ij = int(np.sqrt(num_points) / 10)
    if max_n_ij is not None:
        n_ij = min(n_ij, max_n_ij)  
    return n_ij

def determine_length(r_max, bin_resolution=5):

    length = int(np.log10(r_max) * bin_resolution)  # Bin resolution can be adjusted
    return length

if False:
    def generate_bins(dx, lmax, data, max_n_ij=None, bin_resolution=50):
        """Generate bins for the VSF using automatically determined N_ij and length."""
        N_ij = determine_n_ij(data, max_n_ij=max_n_ij)
        
        r_max = np.max(np.sqrt(data['x']**2 + data['y']**2 + data['z']**2))
        length = determine_length(r_max, bin_resolution=bin_resolution)
        
        return np.linspace(0*r_max, r_max, length)
else:
    def generate_bins(dx, data, max_n_ij=None, bin_resolution=80, length=None):
        N_ij = determine_n_ij(data, max_n_ij=max_n_ij)
        
        # Automatically determine length based on r_max or use provided length
        r_max = np.max(np.sqrt(data['x']**2 + data['y']**2 + data['z']**2))
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

#@njit
if False:
    def compute_vsf(x, y, z, vx, vy, vz, bins_edges, vsf, counts):
        N = len(vx)
        for m in range(N):
            for n in range(m + 1, N):
                dr = np.sqrt((x[m] - x[n])**2 + (y[m] - y[n])**2 + (z[m] - z[n])**2)
                dv = np.sqrt((vx[m] - vx[n])**2 + (vy[m] - vy[n])**2 + (vz[m] - vz[n])**2)

                bin_idx = np.searchsorted(bins_edges, dr) - 1
                if 0 <= bin_idx < len(vsf):
                    vsf[bin_idx] += dv
                    counts[bin_idx] += 1

else: 
    def compute_vsf(x, y, z, vx, vz, bins_edges, vsf, counts):
        sample_size = min(10000, len(x))  # Take the minimum between 10000 and the actual population size
        idx = np.random.choice(len(x), size=sample_size, replace=False)
        x, y, z, vx, vz = x[idx], y[idx], z[idx], vx[idx],  vz[idx]
        coords = np.column_stack((x, y, z))
        vels = np.column_stack((vx, vz))
    
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=bins_edges[-1], output_type='ndarray')
        drs = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        dvs = np.linalg.norm(vels[pairs[:, 0]] - vels[pairs[:, 1]], axis=1)
        
        bin_idx = np.searchsorted(bins_edges, drs) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(vsf))

        np.add.at(vsf, bin_idx[valid], dvs[valid])
        np.add.at(counts, bin_idx[valid], 1)
        
        return vsf, counts


def generate_vsf(data, dx, stand_l, outname, min_pairs=3):
    """Generate the 3D velocity structure function."""
    log_centers = generate_bins(dx, data)
    edges = generate_bins_edges(log_centers)
    bin_edges = 10**edges
    vsf = np.zeros(len(bin_edges) - 1)
    counts = np.zeros_like(vsf)
    x, y, z = data['x'], data['y'], data['z']
    vx, vz = data['vx'], data['vz']
    compute_vsf(x, y, z, vx, vz, bin_edges, vsf, counts)
    #vsf /= np.where(counts > 0, counts, 1)
        # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        vsf = np.where(counts > 0, vsf / counts, np.nan)

    # Mask low-count bins
    vsf = np.where(counts >= min_pairs, vsf, np.nan)
    # Add correction factor for isotropic scaling
    vsf *= 1.5
    np.savez_compressed(outname, vsf=vsf, log_centers=log_centers)

    plt.style.use('custom_plot')
    plt.figure(figsize=(8, 6))

    # Plot VSF against bin centers (log scale for bins)
    plt.plot(10**log_centers/stand_l, vsf, color='blue')
    x_ref = 10**log_centers / stand_l
    # Remove NaNs if needed
    x_ref = x_ref[~np.isnan(vsf)]

    # Pick two points to define the slope line over (scale range to match your data visually)
    x0 = np.min(x_ref[np.isfinite(x_ref)]) * 1.5
    x1 = np.max(x_ref[np.isfinite(x_ref)]) / 4
    x_slope = np.array([x0, x1])
    y_slope = x_slope**(1/3)

    # Scale y_slope to overlay nicely (e.g., match magnitude of vsf visually)
    # You can scale to match roughly the first visible data point
    scale_factor = np.nanmax(vsf) / np.max(y_slope)
    y_slope *= scale_factor

    # Plot the reference slope line
    plt.plot(x_slope, y_slope, 'k--', label=r'$\propto l^{1/3}_{3D}$', linewidth=1)

    # Set plot labels and title
    plt.xlabel('Separation Distance (pc)', fontsize=12)
    plt.ylabel('VSF', fontsize=12)
    plt.title('Velocity Structure Function (VSF)', fontsize=14)

    # Add a grid and legend
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')

    # Display the plot

    plt.savefig(os.path.join(outname+'.png'), dpi=300)
    plt.clf()

    # Save the VSF and bin edges to an HDF5 file
    #with h5py.File(outname+'.h5', 'w') as f:
    #    f.create_dataset('vsf', data=vsf)
    #    f.create_dataset('bins', data=10**log_centers)




def process_run(infile, stand_l, mbar, outdir):
    #try:
    print(infile)
    idx = infile.split('/')[-1].split('.')[-2]
    outfile = os.path.join(outdir, f"noy3D_vsf_{str(int(idx)).zfill(3)}")
    if os.path.exists(outfile + '.png'):
        print(f"[✓] Skipping {outfile}, already exists.")
        return

    data, dx = load_cold_gas_region_and_fields(infile, mbar, n_jobs = 1)
    print("File has been read.")
    generate_vsf(data, dx, stand_l, outfile)
    print(f"[✓] VSF generated: {outfile}")




def run_all_parallel(run_list, stand_l, mbar, outdir, n_procs):
    os.makedirs(outdir, exist_ok=True)
    
    # Use joblib to parallelize process_run for each file
    Parallel(n_jobs=(n_procs//1))(
        delayed(process_run)(infile, stand_l, mbar, outdir)
        for infile in run_list
    )



if __name__ == "__main__":
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")

    
    if True: #len(user_args) == 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
        print('Saved to: ', saveFile)
        if not os.path.exists(os.path.join('/u/ferhi/Figures/velocity_structure_function/',f"{parts[-3]}/{parts[-2]}")): 
            os.makedirs(os.path.join('/u/ferhi/Figures/velocity_structure_function/',f"{parts[-3]}/{parts[-2]}"))


    if False:
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

    print(run_paths)
    single_file_paths = sorted(glob.glob(os.path.join(run_paths[0], 'out/parthenon.prim.[0-9]*.phdf')))

    sim_input = run_paths[0].split('out')[0]
    params = load_params(os.path.join(sim_input, 'ism.in'))
    depth  = float(params['reader'].get('problem/wtopenrun', 'depth'))
    cloud_r  = float(params['reader'].get('problem/wtopenrun', 'r0_cgs'))
    stand_l = 0.1 * cloud_r * unyt.cm * cm_to_pc
    
    
    sim = SingleCloudCC(os.path.join(sim_input, 'ism.in'), dir=sim_input)
    mbar = sim.mbar
    run_all_parallel(single_file_paths, stand_l, mbar, outdir=os.path.join('/u/ferhi/Figures/velocity_structure_function/',saveFile), n_procs=N_procs)
    
