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


# -----------------------------
# IOs assembly functions
# -----------------------------
def reassemble_blocks(block_array):
    """
    Reassemble a blocked array into the full global array.
    
    Parameters:
      block_array: numpy array of shape (nBx, nBy, nBz, nFields, bs1, bs2, bs3)
      
    Returns:
      full_array: numpy array of shape (nFields, nBx*bs1, nBy*bs2, nBz*bs3)
    """
    nBx, nBy, nBz, nFields, bs1, bs2, bs3 = block_array.shape
    ICs_reordered = block_array.transpose(3,0,4,1,5,2,6)
    ICs_full = ICs_reordered.reshape(nFields, nBx*bs1, nBy*bs2, nBz*bs3)
    return ICs_full

def gen_bin(fields, filename):
    
    print(len(fields))    
    ICs = np.stack(fields, axis=3).astype(np.float64)
    save_path = os.path.join(localDir, filename)
    
    with open(save_path, "wb") as f:
       f.write(ICs.tobytes())
    print(f"Saved ICs {ICs.shape} to {save_path} ({os.path.getsize(save_path)} bytes).")
 
    return ICs

def gen_adios(MeshSize, MeshBlockSize, fields, filename):
    
    mbl3, mbl2, mbl1 = MeshBlockSize
    nx3, nx2, nx1 = MeshSize
    nz_blocks, ny_blocks, nx_blocks = (int(nx3/mbl3), int(nx2/mbl2), int(nx1/mbl1))
    x_indices, y_indices, z_indices = np.indices((nx_blocks, ny_blocks, nz_blocks))

    # Flatten the indices to get the logical locations for all blocks at once
    LogicalLocations = np.vstack((x_indices.ravel(), y_indices.ravel(), z_indices.ravel())).T
    n_blocks = LogicalLocations.shape[0]

    # Pre-allocate block data
    block_data = np.zeros((n_blocks, len(fields), mbl3, mbl2, mbl1), dtype=np.float64)
    
    meshblock_fields = []
    for meshblock_field in fields:
        meshblock_fields.append(meshblock_field.reshape(nx_blocks, mbl3, ny_blocks, mbl2, nz_blocks, mbl1))


    for i, (loc_x, loc_y, loc_z) in enumerate(LogicalLocations):
        for f in range(len(fields)):
            block_data[i, f, :, :, :] = meshblock_fields[f][loc_x, :, loc_y, :, loc_z, :]

        
    ICs = block_data.reshape(nz_blocks, ny_blocks, nx_blocks, len(fields), mbl3, mbl2, mbl1)
    saveDir = os.path.join(localDir, filename)
    shape = ICs.shape # .tolist()
    start = np.zeros_like(shape).tolist()
    count = ICs.shape #.tolist()
    nsteps = 1
    
    with Stream(saveDir, "w") as s:
        for _ in s.steps(nsteps):
            s.write(filename.split('.bp')[0], ICs, shape, start, count)
    
    print(f"Saved 4D array {ICs.shape} to {saveDir}. Size: {os.path.getsize(saveDir)} bytes.")
    ICs_correct = reassemble_blocks(ICs)
    return ICs_correct.reshape(len(fields), nx3, nx2, nx1)


def load_params(filename_input):
    """ Load simulation parameters from the input file. """
    reader = ut.AthenaPKInputFileReader(filename_input)
    mesh_keys = ['nx1', 'nx2', 'nx3', 'x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max']
    mesh_params = {key: float(reader.get('parthenon/mesh', key)) for key in mesh_keys}
    
    problem_keys = [ 'T_base', 'a_over_H', 'surface_density']
    problem_params = {key: float(reader.get('problem/stratified_box', key)) for key in problem_keys}
    
    gamma = float(reader.get('hydro', 'gamma'))
    
    return {**mesh_params, **problem_params, 'gamma': gamma, 'reader': reader}

def generate_ICs(filename_input, filename='ICs.bp'):
    params = load_params(filename_input)
    stratified_box = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    mbar_over_kb = stratified_box.mbar/ut.constants.kb 
    code_length_cgs = float(params['reader'].get('units', 'code_length_cgs'))
    code_mass_cgs = float(params['reader'].get('units', 'code_mass_cgs'))

    g0 = 2 * np.pi * ut.constants.G * params['surface_density'] * code_mass_cgs / (code_length_cgs)**2
    c_s = np.sqrt(params['T_base'] / mbar_over_kb)
    H = c_s**2/ g0
    rho_0 = params['surface_density'] * code_mass_cgs / (code_length_cgs)**2 \
        / (math.sqrt(2*math.pi) * H)
    print(f"c_s = {c_s/1e5:.3e} km/s")
    print(f"Using rho_0 = {rho_0:.3e} g/cm^3, H = {H/code_length_cgs:.3e} code units, g0 = {g0:.3e} cm/s^2")


    full_box_rho = isothermal_strat(nx1, nx2, nx3, rho_0, params['a_over_H'], H,
                    (params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                    (params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                    (params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs)
                    )
    mom = np.zeros_like(full_box_rho)
    en = 0.5 * mom**2 / full_box_rho +  params['T_base'] / mbar_over_kb * full_box_rho / (params['gamma'] - 1)
    fields = (full_box_rho, mom, en)
    
    MeshBlockSize = (mbl3, mbl2, mbl1)
    MeshSize = (nx3, nx2, nx1)

    if filename.split(".")[-1] == "bin":
        ICs = gen_bin(fields, filename)
    elif filename.split(".")[-1] == "bp":
        ICs = gen_adios(MeshSize, MeshBlockSize, fields, filename)

        
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

def isothermal_strat(nx, ny, nz, rho0, a, H, x_range, y_range, z_range):

    # Create the 3D grid
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Compute the density
    rho = rho0 * np.exp(-a * (np.sqrt(1 + (Y / (a * H))**2) - 1))

    return rho


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate stratified box.")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of parallel jobs.")
    args = parser.parse_args()

    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'strat.in')
    generate_ICs(filename_input=filename_input, filename='ICs.bp')
