import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pyFC
import math
from scipy.ndimage import label, gaussian_filter
import utils as ut
from adjust_ics import *
import sys
from joblib import Parallel, delayed
from adios2 import Stream
from scipy.special import k1
from adioslib import *





def load_params(filename_input):
    """ Load simulation parameters from the input file. """
    reader = ut.AthenaPKInputFileReader(filename_input)
    mesh_keys = ['nx1', 'nx2', 'nx3', 'x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max']
    mesh_params = {key: float(reader.get('parthenon/mesh', key)) for key in mesh_keys}
    
    problem_keys = [ 'T_base', 'a_over_H', 'surface_density', 'r_cloud_inserted', 'T_cloud']
    problem_params = {key: float(reader.get('problem/stratified_box', key)) for key in problem_keys}
    
    gamma = float(reader.get('hydro', 'gamma'))
    
    return {**mesh_params, **problem_params, 'gamma': gamma, 'reader': reader}

def generate_ICs(filename_input, filename='ICs.bp', localDir='.'):

    full_box_rho = gen_rho_strat(filename_input)
    

    params = load_params(filename_input)
    stratified_box = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    mbar_over_kb = stratified_box.mbar/ut.constants.kb 
    code_length_cgs = float(params['reader'].get('units', 'code_length_cgs'))
    code_mass_cgs = float(params['reader'].get('units', 'code_mass_cgs'))

    mom1 = np.zeros_like(full_box_rho); mom2 = np.zeros_like(full_box_rho); mom3 = np.zeros_like(full_box_rho)
    en = 0.5 * mom**2 / full_box_rho +  params['T_base'] / mbar_over_kb * full_box_rho / (params['gamma'] - 1)

    # Insert cloud
    rho_with_cloud, mom1_with_cloud, mom2_with_cloud, mom3_with_cloud, en_with_cloud = insert_sphere(full_box_rho, mom1, mom2, mom3, en, r=params['r_cloud_inserted'] * code_length_cgs, 
                            T_cloud=params['T_cloud'], 
                            mbar_over_kb=mbar_over_kb, gamma=params['gamma'], T_base = params['T_base'],
                            x_range=(params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                            y_range=(params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                            z_range=(params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs),
                            inplace=False)
    
    fields_ICs = (rho_with_cloud, mom1_with_cloud, mom2_with_cloud, mom3_with_cloud, en_with_cloud)

    
    MeshBlockSize = (mbl3, mbl2, mbl1)
    MeshSize = (nx3, nx2, nx1)


    if filename.split(".")[-1] == "bin":
        ICs = gen_bin(fields_ICs, filename)
    elif filename.split(".")[-1] == "bp":
        ICs = gen_adios(MeshSize, MeshBlockSize, fields_ICs, filename, localDir=localDir)

        
    print(f"ICs shape: {ICs.shape}")
    try:
        plt.imshow(ICs[0, :, :, nx3 // 2], cmap='viridis', norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.savefig("ICs_slice.png")
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}. Maybe you have selected the wrong data type for ICs?")


    return ICs


# -----------------------------
# ICs set-up
# -----------------------------
def gen_init_params(params, code_mass_cgs, code_length_cgs, mbar_over_kb):
    g0 = 2 * np.pi * ut.constants.G * params['surface_density'] * code_mass_cgs / (code_length_cgs)**2
    c_s = np.sqrt(params['T_base'] / mbar_over_kb)
    H = c_s**2/ g0
    rho_0 = (params['surface_density'] * code_mass_cgs / code_length_cgs**2
    ) / (2 * H)
    print(f"c_s = {c_s/1e5:.3e} km/s")
    print(f"Using rho_0 = {rho_0:.3e} g/cm^3, H = {H/code_length_cgs:.3e} code units, g0 = {g0:.3e} cm/s^2")

    return g0, H, rho_0

def find_y_for_density(rho_target, rho0, a, H):
    """
    Find the y-coordinate where density equals rho_target in an isothermal stratification.
    
    Solves: rho_target = rho0 * exp(-a * (sqrt(1 + (y/(a*H))^2) - 1))
    """
    if rho_target >= rho0:
        return 0.0
    
    # Rearrange: sqrt(1 + (y/(a*H))^2) = 1 - ln(rho_target/rho0)/a
    term = 1 - np.log(rho_target / rho0) / a
    
    if term <= 1:
        return 0.0
    
    # y = a*H * sqrt(term^2 - 1)
    y = a * H * np.sqrt(term**2 - 1)
    return y

def isothermal_strat(nx, ny, nz, rho0, a, H, x_range, y_range, z_range):
    # Create the 3D grid
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Compute the density
    rho = rho0 * np.exp(-a * (np.sqrt(1 + (Y / (a * H))**2) - 1))
    return rho

def gen_rho_strat(filename_input):
    params = load_params(filename_input)
    stratified_box = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbar_over_kb = stratified_box.mbar/ut.constants.kb 
    code_length_cgs = float(params['reader'].get('units', 'code_length_cgs'))
    code_mass_cgs = float(params['reader'].get('units', 'code_mass_cgs'))
    
    # Get initial parameters
    g0, H, rho_0 = gen_init_params(params, code_mass_cgs, code_length_cgs, mbar_over_kb=mbar_over_kb)
    
    # Calculate target density at boundaries
    rho_target = 0.0001 * stratified_box.mbar 
    
    # Find y-coordinate where density equals rho_target (positive side)
    y_boundary = find_y_for_density(rho_target, rho_0, params['a_over_H'], H)
    
    # Calculate cloud position: 10*r_cloud from upper boundary
    r_cloud_cgs = params['r_cloud_inserted'] * code_length_cgs
    
    # Total domain size needed so that:
    # x2max = y_boundary and is at 0.9 of total domain
    # This means: y_boundary = 0.9 * total_domain
    total_domain = y_boundary / 0.9
    
    # Set new x2min and x2max in code units
    x2max_new = 0.9 * total_domain / code_length_cgs
    x2min_new = -0.1 * total_domain / code_length_cgs
    
    # Verify cloud position is valid (10*r from x2max)
    y_cloud_cgs = (x2max_new * code_length_cgs) - 10 * r_cloud_cgs
    y_cloud = y_cloud_cgs / code_length_cgs
    
    print(f"Adjusting x2 boundaries:")
    print(f"  Old x2min: {params['x2min']:.6f}, Old x2max: {params['x2max']:.6f}")
    print(f"  New x2min: {x2min_new:.6f}, New x2max: {x2max_new:.6f}")
    print(f"  Total domain size: {total_domain/code_length_cgs:.6f} (code units)")
    print(f"  x2max/total: {x2max_new/(total_domain/code_length_cgs):.2f}, x2min/total: {x2min_new/(total_domain/code_length_cgs):.2f}")
    print(f"  Boundary density at x2max: {rho_target:.6e} (code units)")
    print(f"  Cloud position: y = {y_cloud:.6f} (code units), distance from x2max: {(x2max_new - y_cloud)*code_length_cgs/r_cloud_cgs:.1f}*r")
    
    # Update the reader
    stratified_box.reader.set_('parthenon/mesh', 'x2min', x2min_new)
    stratified_box.reader.set_('parthenon/mesh', 'x2max', x2max_new)
    
    # Update params dictionary
    params['x2min'] = x2min_new
    params['x2max'] = x2max_new
    
    
    full_box_rho = isothermal_strat(nx1, nx2, nx3, rho_0, params['a_over_H'], H,
                        (params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                        (params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                        (params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs))
    return full_box_rho

def insert_sphere(density, mom1, mom2, mom3, energy,
                  r,
                  T_cloud,
                  mbar_over_kb,
                  gamma,
                  T_base,
                  x_range,
                  y_range,
                  z_range,
                  inplace=True, 
                  return_cloud_rho=False):
    if density.shape != energy.shape:
        raise ValueError("density and energy must have the same shape (ny, nx, nz)")
    nx, ny, nz = density.shape
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    x_center = 0.5 * (x_range[0] + x_range[1])
    z_center = 0.5 * (z_range[0] + z_range[1])
    
    # Place cloud 10*r from upper boundary
    y_center = y_range[1] - 10 * r
    
    print(f'Cloud of radius {r/((y_range[1]-y_range[0])/ny):.3f} cells inserted at ({x_center}, {y_center}, {z_center})')
    
    if r <= 0:
        print("WARNING: r must be positive. Defaulting to no cloud.")
        return density, mom1, mom2, mom3, energy
    if not (y_range[0] <= y_center <= y_range[1]):
        raise ValueError("Computed y_center is outside the provided y_range."
                       " Check r and y_range values.")
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mask = (X - x_center)**2 + (Y - y_center)**2 + (Z - z_center)**2 <= r**2
    
    if not inplace:
        density = density.copy()
        energy = energy.copy()
    
    rho_cloud = np.average(density[mask]) * T_base / T_cloud
    
    if return_cloud_rho:
        return rho_cloud, np.average(density[mask]), y_center
    else:
        density[mask] = rho_cloud
        mom1[mask] = mom2[mask] = mom3[mask] = 0.0
        energy_value = (T_cloud / mbar_over_kb) * rho_cloud / (gamma - 1.0)
        energy[mask] = energy_value
        return density, mom1, mom2, mom3, energy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate stratified box.")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of parallel jobs.")
    args = parser.parse_args()

    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'strat.in')
    generate_ICs(filename_input=filename_input, filename='ICs.bp', localDir=localDir)
