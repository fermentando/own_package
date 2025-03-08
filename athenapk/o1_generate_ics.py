import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pyFC
import math
import utils as ut
from scipy.ndimage import gaussian_filter


localDir = os.getcwd()


def load_params(filename_input):
    reader = ut.AthenaPKInputFileReader(filename_input)
    
    nx1 = int(reader.get('parthenon/mesh', 'nx1'))
    nx2 = int(reader.get('parthenon/mesh', 'nx2'))
    nx3 = int(reader.get('parthenon/mesh', 'nx3'))
    
    xmin1 = float(reader.get('parthenon/mesh', 'x1min'))
    xmax1 = float(reader.get('parthenon/mesh', 'x1max'))
    xmin2 = float(reader.get('parthenon/mesh', 'x2min'))
    xmax2 = float(reader.get('parthenon/mesh', 'x2max'))
    xmin3 = float(reader.get('parthenon/mesh', 'x3min'))
    xmax3 = float(reader.get('parthenon/mesh', 'x3max'))
    
    rho_cloud = float(reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
    rho_wind = float(reader.get('problem/wtopenrun', 'rho_wind_cgs'))
    T_wind = float(reader.get('problem/wtopenrun', 'T_wind_cgs'))
    gamma = float(reader.get('hydro', 'gamma'))

    Rcloud = float(reader.get('problem/wtopenrun', 'r0_cgs')) / float(reader.get('units', 'code_length_cgs'))
    
    return {
        'reader': reader,
        'nx1': nx1, 'nx2': nx2, 'nx3': nx3,
        'xmin1': xmin1, 'xmax1': xmax1,
        'xmin2': xmin2, 'xmax2': xmax2,
        'xmin3': xmin3, 'xmax3': xmax3,
        'rho_cloud': rho_cloud, 'rho_wind': rho_wind,
        'T_wind': T_wind, 'gamma': gamma, 
        'Rcloud': Rcloud
    }


def compute_wind_velocity(params):
    reader = params['reader']
    rho_wind = params['rho_wind']
    T_wind = params['T_wind']
    gamma = params['gamma']
    
    try:
        v_wind = float(reader.get('problem/wtopenrun', 'v_wind_cgs'))
    except:
        Mach_wind = float(reader.get('problem/wtopenrun', 'Mach_wind'))
        He_mass_fraction = float(reader.get('hydro', 'He_mass_fraction'))        
        mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2)
        mean_mol_mass = mu * ut.constants.uam
        v_wind = np.sqrt(gamma * ut.constants.kb * T_wind / mean_mol_mass) * Mach_wind
    
    return v_wind


def create_singlecloud(filename_input):
    params = load_params(filename_input)
    
    gamma = params['gamma']
    gm1 = gamma - 1.
    v_wind = compute_wind_velocity(params)
    rhoe_wind = params['rho_wind'] * params['T_wind'] / gm1
    
    x = np.linspace(params['xmin1'], params['xmax1'], params['nx1'])
    y = np.linspace(params['xmin2'], params['xmax2'], params['nx2'])
    z = np.linspace(params['xmin3'], params['xmax3'], params['nx3'])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


    distance_from_center = np.sqrt((X)**2 + (Z)**2)
    Rcloud = params['Rcloud']
    full_box_rho = np.where(distance_from_center <= Rcloud, params['rho_cloud'], params['rho_wind'])
    
    mom = np.zeros_like(full_box_rho)
    en1 = 0.5 * mom * mom / full_box_rho
    en2 = np.ones_like(mom) * rhoe_wind
    
    ICs = np.stack((full_box_rho, mom, en1, en2), axis=3)
    ICs = np.array(ICs, dtype=np.float64)
    
    saveDir = os.path.abspath(os.path.join(localDir, 'ICs.bin'))
    with open(saveDir, "wb") as f:
        f.write(ICs.tobytes())
    
    print(f"4D array of shape {ICs.shape} saved to {saveDir}.")
    print(f"File size: {os.path.getsize(saveDir)} bytes.")


def pyFC_create_FractalISM(kmax=None, ism_width_ratio = 8, mean=10, sigma=5, filename_input='bin_cloud.in'):
    params = load_params(filename_input)
    reader = params['reader']
    fv = float(reader.get('problem/wtopenrun', 'fv'))
    in_kmin = reader.get('problem/wtopenrun', 'kmin')
    nx2 = params['nx2']; nx1 = params['nx1']; nx3 = params['nx3'] 

    R_ism_width = 2 * params['Rcloud']/((params['xmax2'] - params['xmin2'])/ism_width_ratio)
    k_cloud =  nx2 / nx1 / R_ism_width 

    if in_kmin.replace(".", "", 1).isdigit():
        in_kmin_float = float(in_kmin)
        if in_kmin_float >= k_cloud:
            kmin = in_kmin_float * k_cloud
            print(f"Assigned kmin is {kmin / k_cloud:.2f} kcloud.")
        else:
            print("WARNING: K fractal value smaller than accepted, setting to width default.")
            kmin = (nx1 / nx2) * ism_width_ratio
    elif in_kmin.lower() in ["max", "maximum"]:
        kmin = (nx1 / nx2) * ism_width_ratio
        print(f"Assigned kmin is {kmin / k_cloud:.2f} kcloud.")
    else:
        kmin = k_cloud
        print(f"Assigned kmin is set to default: {kmin / k_cloud:.2f} kcloud.")


    ism_cube = pyFC.LogNormalFractalCube(nx1, math.ceil(nx2/ism_width_ratio), nx3, kmin, kmax, mean, sigma).gen_cube()
    rho_fluct = ism_cube.cube.astype(dtype=np.int64)
    
    vol_frac_lim = int((1 - fv) * (nx1 * int(nx2 / ism_width_ratio) * nx3))
    rho_threshold = rho_fluct.flatten()[np.argsort(rho_fluct.flatten())[vol_frac_lim - 1]]
    rho_field = np.where(rho_fluct <= rho_threshold, params['rho_wind'], params['rho_cloud'])
    full_box_rho = np.ones((nx1, nx2, nx3)) * params['rho_wind']
     

    # Compute the start and end indices where rho_field will be placed
    start_idx = int(nx2/10)
    end_idx = start_idx + rho_field.shape[1]
    full_box_rho[:, start_idx:end_idx, :] = rho_field
    
    mom = np.zeros_like(full_box_rho)
    en1 = 0.5 * mom * mom / full_box_rho
    en2 = np.ones_like(mom) * (params['rho_wind'] * params['T_wind'] / (params['gamma'] - 1))
    
    ICs = np.stack((full_box_rho, mom, en1, en2), axis=3)
    ICs = np.array(ICs, dtype=np.float64)

    plt.imshow(ICs[:, :, nx3 // 2, 0], cmap='viridis', norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.show()
    
    saveDir = os.path.abspath(os.path.join(localDir, 'ICs.bin'))
    with open(saveDir, "wb") as f:
        f.write(ICs.tobytes())
    
    print(f"4D array of shape {ICs.shape} saved to {saveDir}.")

# Simulate a percolation field using a threshold and apply a Gaussian filter
def simulate_percolation_once(dimensions, p, sigma):
    """
    Simulate one percolation instance with the given parameters.
    Returns the percolation field for a non-cubic 3D grid.
    
    :param dimensions: A tuple representing the 3D dimensions of the grid (nx1, nx2, nx3).
    :param p: The percolation probability (fraction of the grid that is considered dense).
    :param sigma: The standard deviation for Gaussian smoothing.
    :return: A 3D percolation field.
    """
    field = np.random.rand(*dimensions)
    smoothed_field = gaussian_filter(field.astype(float), sigma=sigma)
    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    percolation_field = smoothed_field > threshold
    return percolation_field

def create_ISM(ism_width_ratio=4/3, filename_input='ism.in'):
    # Load parameters (assuming `load_params` function exists)
    params = load_params(filename_input)
    reader = params['reader']
    
    # ISM field parameters
    nx1, nx2, nx3 = params['nx1'], params['nx2'], params['nx3']
    cell_size = (params['xmax1']-params['xmin1'])/nx1
    
    
    # Get kin from the reader
    kin = float(reader.get('problem/wtopenrun', 'kmin'))
    
    # Get Rcloud from the params
    Rcloud = params['Rcloud']
    
    # Calculate sigma using the given formula
    sigma = kin * Rcloud / 6 / cell_size
    print(sigma, Rcloud/cell_size)
    
    # Get fv (volume fraction) from reader
    fv = float(reader.get('problem/wtopenrun', 'fv'))
    
    # Percolation-based ISM generation
    percolation_field = simulate_percolation_once((nx1, math.ceil(nx2/ism_width_ratio), nx3), fv, sigma)  # Generate percolation field
    rho_fluct = percolation_field.astype(np.int64)

    # Set threshold and apply density values (rho_cloud and rho_wind)
    vol_frac_lim = int((1 - fv) * (nx1 * int(nx2 / ism_width_ratio) * nx3))
    rho_threshold = rho_fluct.flatten()[np.argsort(rho_fluct.flatten())[vol_frac_lim - 1]]
    rho_field = np.where(rho_fluct <= rho_threshold, params['rho_wind'], params['rho_cloud'])

    # Initialize full box with rho_wind
    full_box_rho = np.ones((nx1, nx2, nx3)) * params['rho_wind']

    # Place rho_field into full box in a section
    start_idx = int(nx2 / 10)
    end_idx = start_idx + rho_field.shape[1]
    full_box_rho[:, start_idx:end_idx, :] = rho_field

    # Initialize momentum, energy, and other fields
    mom = np.zeros_like(full_box_rho)
    en1 = 0.5 * mom * mom / full_box_rho  # Kinetic energy: 0.5 * rho * v^2, v=0 so this is initially 0
    en2 = np.ones_like(mom) * (params['rho_wind'] * params['T_wind'] / (params['gamma'] - 1))  # Internal energy

    # Stack the fields (rho, momentum, kinetic energy, internal energy)
    ICs = np.stack((full_box_rho, mom, en1, en2), axis=3)
    ICs = np.array(ICs, dtype=np.float64)

    # Plot the rho field (density) slice at nx3 // 2
    plt.imshow(ICs[:, :, nx3 // 2, 0], cmap='viridis', norm=matplotlib.colors.LogNorm())
    plt.colorbar(label='Density')
    plt.title("Initial Density Field in ISM")
    plt.show()


    saveDir = os.path.abspath(os.path.join(localDir, 'ICs.bin'))
    with open(saveDir, "wb") as f:
        f.write(ICs.tobytes())

    print(f"4D array of shape {ICs.shape} saved to {saveDir}.")

if __name__ == "__main__":
    
    if len(sys.argv) > 1: width = float(sys.argv[1])
    else: width = 4
    filename_input = os.path.join(localDir, 'ism.in')
    
    #create_singlecloud(filename_input)
    create_ISM(filename_input=filename_input)

