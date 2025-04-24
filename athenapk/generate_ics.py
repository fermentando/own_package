import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pyFC
import math
from scipy.ndimage import gaussian_filter
import utils as ut
import sys
from joblib import Parallel, delayed
from adios2 import Stream

# -------- Parameter Reading and Initial Conditions Output -------- #

def load_params(filename_input):
    """ Load simulation parameters from the input file. """
    reader = ut.AthenaPKInputFileReader(filename_input)
    mesh_keys = ['nx1', 'nx2', 'nx3', 'x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max']
    mesh_params = {key: float(reader.get('parthenon/mesh', key)) for key in mesh_keys}
    
    problem_keys = ['rho_cloud_cgs', 'rho_wind_cgs', 'T_wind_cgs', 'r0_cgs']
    problem_params = {key: float(reader.get('problem/wtopenrun', key)) for key in problem_keys}
    
    gamma = float(reader.get('hydro', 'gamma'))
    Rcloud = problem_params['r0_cgs'] / float(reader.get('units', 'code_length_cgs'))
    
    return {**mesh_params, **problem_params, 'gamma': gamma, 'Rcloud': Rcloud, 'reader': reader}


def compute_wind_velocity(params):
    """ Compute wind velocity based on input parameters. """
    reader = params['reader']
    gamma = params['gamma']
    T_wind = params['T_wind_cgs']
    rho_wind = params['rho_wind_cgs']
    try:
        return float(reader.get('problem/wtopenrun', 'v_wind_cgs'))
    except:
        Mach_wind = float(reader.get('problem/wtopenrun', 'Mach_wind'))
        He_frac = float(reader.get('hydro', 'He_mass_fraction'))
        mu = 1 / (He_frac * 3 / 4 + (1 - He_frac) * 2)
        mean_mol_mass = mu * ut.constants.uam
        return np.sqrt(gamma * ut.constants.kb * T_wind / mean_mol_mass) * Mach_wind


def generate_ICs(params, rho_field, filename='ICs.bp'):
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    full_box_rho = np.ones((nx3, nx2, nx1)) * params['rho_wind_cgs']
    start_idx = nx2 // 10 #(nx2 - rho_field.shape[1])//2
    full_box_rho[:, start_idx:start_idx + rho_field.shape[1], :] = rho_field
    
    mom = np.zeros_like(full_box_rho)
    en1 = 0.5 * mom**2 / full_box_rho
    en2 = np.ones_like(mom) * params['rho_wind_cgs'] * params['T_wind_cgs'] / (params['gamma'] - 1)
    ICs = np.stack((full_box_rho, mom, en1, en2), axis=3).astype(np.float64)
    
    saveDir = os.path.join(localDir, filename)
    shape = ICs.shape # .tolist()
    start = np.zeros_like(shape).tolist()
    count = ICs.shape #.tolist()
    nsteps = 1
    
    with Stream(saveDir, "w") as s:
        for _ in s.steps(nsteps):
            s.write(filename.split('.bp')[0], ICs, shape, start, count)
    
    print(f"Saved 4D array {ICs.shape} to {saveDir}. Size: {os.path.getsize(saveDir)} bytes.")
    return ICs

# -------- Single Cloud Generation -------- #
def generate_sphere(filename_input, filename='ICs.bin'):
    params = load_params(filename_input)
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    x1min, x1max = params['x1min'], params['x1max']
    x2min, x2max = params['x2min'], params['x2max']
    x3min, x3max = params['x3min'], params['x3max']
    Rcloud = params['Rcloud']
    
    x1 = np.linspace(x1min, x1max, nx1)
    x2 = np.linspace(x2min, x2max, nx2)
    x3 = np.linspace(x3min, x3max, nx3)
    
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    
    # Compute the distance from the origin
    distance = np.sqrt(X1**2 + X2**2 + X3**2)
    
    # Create a 3D array with values of rho_cloud and rho_empty
    full_box_rho = np.where(distance <= Rcloud, params['rho_cloud_cgs'], params['rho_wind_cgs'])
    
    mom = np.zeros_like(full_box_rho)
    en1 = 0.5 * mom**2 / full_box_rho
    en2 = np.ones_like(mom) * params['rho_wind_cgs'] * params['T_wind_cgs'] / (params['gamma'] - 1)
    ICs = np.stack((full_box_rho, mom, en1, en2), axis=3).astype(np.float64)
    
    plt.imshow(ICs[:, :, nx3 // 2, 0], cmap='viridis', norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.savefig("ICs_slice.png")
    plt.show()
    
    save_path = os.path.join(localDir, filename)
    with open(save_path, "wb") as f:
        f.write(ICs.tobytes())

    print(f"Saved ICs {ICs.shape} to {save_path} ({os.path.getsize(save_path)} bytes).")
    return ICs
    
    
# -------- Fractal ISM and Percolation Generation -------- #

def simulate_percolation(dimensions, p, sigma):
    """ Simulate percolation with Gaussian smoothing. """
    field = np.random.rand(*dimensions)
    smoothed_field = gaussian_filter(field, sigma=sigma) if not isinstance(sigma, tuple) else np.mean(
        [gaussian_filter(field, s) for s in np.linspace(*sigma, 5)], axis=0)
    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    return smoothed_field > threshold


def create_ISM(filename_input='ism.in', ism_depth=1, fv=None, n_jobs=1):
    """ Generate ISM field with percolation and density fluctuations. """
    params = load_params(filename_input)
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    cell_size = (params['x2max'] - params['x2min']) / nx2
    Rcloud = params['Rcloud']
    fv = fv or float(params['reader'].get('problem/wtopenrun', 'fv'))
    kin = params['reader'].get('problem/wtopenrun', 'kmin')
    ism_depth = params['reader'].get('problem/wtopenrun', 'depth')
    sigma = float(kin) * Rcloud / 10 / cell_size if ',' not in kin else tuple(float(k) * Rcloud / 10 / cell_size for k in kin.split(','))
    
    print("Sigma:", sigma)
    dimensions = (nx1, math.ceil(float(ism_depth) * Rcloud/cell_size), nx3)
    
    # Parallel percolation simulation
    percolation_fields = Parallel(n_jobs=n_jobs)(delayed(simulate_percolation)(dimensions, fv, sigma) for _ in range(1))
    percolation_field = percolation_fields[0]

    rho_field = np.where(percolation_field, params['rho_cloud_cgs'], params['rho_wind_cgs'])

    ICs = generate_ICs(params, rho_field)
    plt.imshow(ICs[:, :, nx3 // 2, 0], cmap='viridis', norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.savefig("ICs_slice.png")
    plt.show()

    return percolation_field, sigma

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ISM with parallel percolation.")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument('-r', type=float, default=1, help="Width of the ISM slab in r_cloud.")
    args = parser.parse_args()

    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'ism.in')
    #generate_sphere(filename_input=filename_input)
    create_ISM(filename_input=filename_input, n_jobs=args.n_jobs, ism_depth=args.r)

