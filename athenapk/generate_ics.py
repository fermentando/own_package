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
import adios2

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

def gen_adios_streaming(MeshSize, MeshBlockSize, fields, filename):
    """
    Write mesh-blocked field data to ADIOS without large intermediate arrays.

    Final ADIOS variable shape:
        (nz_blocks, ny_blocks, nx_blocks,
         n_fields, mbl3, mbl2, mbl1)
    """

    # Unpack sizes
    mbl3, mbl2, mbl1 = MeshBlockSize
    nx3, nx2, nx1 = MeshSize

    nz_blocks = nx3 // mbl3
    ny_blocks = nx2 // mbl2
    nx_blocks = nx1 // mbl1
    n_fields = len(fields)

    # Sanity checks
    assert nx3 % mbl3 == 0
    assert nx2 % mbl2 == 0
    assert nx1 % mbl1 == 0

    # Reshape fields into block views (no copies if contiguous)
    # Original field shape: (nx3, nx2, nx1)
    # Target shape: (nz_blocks, mbl3, ny_blocks, mbl2, nx_blocks, mbl1)
    meshblock_fields = [
        field.reshape(nz_blocks, mbl3,
                      ny_blocks, mbl2,
                      nx_blocks, mbl1)
        for field in fields
    ]

    saveDir = os.path.join(localDir, filename)
    varname = filename.split(".bp")[0]

    # Global shape of the ADIOS variable
    global_shape = (
        nz_blocks,
        ny_blocks,
        nx_blocks,
        n_fields,
        mbl3,
        mbl2,
        mbl1
    )

    with Stream(saveDir, "w") as s:
        for _ in s.steps(1):

            for bx in range(nx_blocks):
                for by in range(ny_blocks):
                    for bz in range(nz_blocks):

                        # Build one block: shape (n_fields, mbl3, mbl2, mbl1)
                        block = np.empty(
                            (n_fields, mbl3, mbl2, mbl1),
                            dtype=fields[0].dtype
                        )

                        for f in range(n_fields):
                            block[f] = meshblock_fields[f][
                                bz, :, by, :, bx, :
                            ]

                        # ADIOS expects full dimensionality
                        block = block[np.newaxis, np.newaxis, np.newaxis, ...]

                        start = (
                            bz,
                            by,
                            bx,
                            0,
                            0,
                            0,
                            0
                        )

                        count = (
                            1,
                            1,
                            1,
                            n_fields,
                            mbl3,
                            mbl2,
                            mbl1
                        )

                        s.write(
                            varname,
                            block,
                            shape=global_shape,
                            start=start,
                            count=count
                        )

    # Optional: reconstruct original field layout for return
    # This allocates full memory again – remove if not needed
    #reconstructed = np.empty(
    #    (n_fields, nx3, nx2, nx1),
    #    dtype=fields[0].dtype
    #)

    #for f in range(n_fields):
    #    reconstructed[f] = fields[f]

    return None
def gen_adios_streaming_new(MeshSize, MeshBlockSize, fields, filename):
    """
    Write mesh-blocked field data to ADIOS with memory-efficient step-based flushing.
    Uses multiple steps to allow ADIOS2 to flush buffers naturally.
    """
    mbl3, mbl2, mbl1 = MeshBlockSize
    nx3, nx2, nx1 = MeshSize

    nz_blocks = nx3 // mbl3
    ny_blocks = nx2 // mbl2
    nx_blocks = nx1 // mbl1
    n_fields = len(fields)

    assert nx3 % mbl3 == 0
    assert nx2 % mbl2 == 0
    assert nx1 % mbl1 == 0

    saveDir = os.path.join(localDir, filename)
    varname = filename.split(".bp")[0]

    global_shape = (
        nz_blocks,
        ny_blocks,
        nx_blocks,
        n_fields,
        mbl3,
        mbl2,
        mbl1
    )

    total_blocks = nx_blocks * ny_blocks * nz_blocks
    blocks_per_step = 20  # Process 20 blocks per step to avoid buffering
    
    # Precompute all block coordinates
    block_coords = []
    for bx in range(nx_blocks):
        for by in range(ny_blocks):
            for bz in range(nz_blocks):
                block_coords.append((bx, by, bz))
    
    # Calculate number of steps needed
    n_steps = (total_blocks + blocks_per_step - 1) // blocks_per_step
    
    with Stream(saveDir, "w") as s:
        for step_idx, _ in enumerate(s.steps(n_steps)):
            step_start = step_idx * blocks_per_step
            step_end = min((step_idx + 1) * blocks_per_step, total_blocks)
            
            for idx in range(step_start, step_end):
                bx, by, bz = block_coords[idx]
                
                # Build one block: shape (n_fields, mbl3, mbl2, mbl1)
                block = np.empty(
                    (n_fields, mbl3, mbl2, mbl1),
                    dtype=fields[0].dtype
                )

                # Extract from each field individually
                for f in range(n_fields):
                    z_start = bz * mbl3
                    z_end = z_start + mbl3
                    y_start = by * mbl2
                    y_end = y_start + mbl2
                    x_start = bx * mbl1
                    x_end = x_start + mbl1
                    
                    block[f] = fields[f][z_start:z_end, y_start:y_end, x_start:x_end]

                start = [bz, by, bx, 0, 0, 0, 0]
                count = [1, 1, 1, n_fields, mbl3, mbl2, mbl1]

                s.write(
                    varname,
                    block,
                    shape=global_shape,
                    start=start,
                    count=count
                )
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{total_blocks} blocks")

    print(f"ICs written to {saveDir}")
    return None

def gen_adios_chunked(MeshSize, MeshBlockSize, fields, filename):
    """
    Memory-efficient: write as single variable using ADIOS2 block writes
    """
    mbl3, mbl2, mbl1 = MeshBlockSize
    nx3, nx2, nx1 = MeshSize
    nz_blocks, ny_blocks, nx_blocks = (nx3 // mbl3, nx2 // mbl2, nx1 // mbl1)
    n_fields = len(fields)
    
    # Global shape
    global_shape = (nz_blocks, ny_blocks, nx_blocks, n_fields, mbl3, mbl2, mbl1)
    
    saveDir = os.path.join(localDir, filename)
    var_name = filename.split('.bp')[0]
    
    # Use lower-level API for block writing
    adios = adios2.Adios()
    io = adios.declare_io("writer")
    
    # Define variable with global shape
    var = io.define_variable(var_name, np.array([]), global_shape, [0]*7, global_shape)
    
    # Mode is an integer: 2 = Write
    from adios2.bindings import Mode
    engine = io.open(saveDir, Mode.Write)
    
    # Reshape fields
    meshblock_fields = [f.reshape(nx_blocks, mbl1, ny_blocks, mbl2, nz_blocks, mbl3) 
                        for f in fields]
    
    # Write blocks
    x_indices, y_indices, z_indices = np.indices((nx_blocks, ny_blocks, nz_blocks))
    LogicalLocations = np.vstack((x_indices.ravel(), y_indices.ravel(), z_indices.ravel())).T
    
    engine.begin_step()
    
    for ix, iy, iz in LogicalLocations:
        block_data = np.zeros((n_fields, mbl3, mbl2, mbl1), dtype=np.float32)
        for f in range(n_fields):
            block_data[f] = meshblock_fields[f][ix, :, iy, :, iz, :]
        
        # Set selection for this block
        start = [iz, iy, ix, 0, 0, 0, 0]
        count = [1, 1, 1, n_fields, mbl3, mbl2, mbl1]
        var.set_selection([start, count])
        
        engine.put(var, block_data)
    
    engine.end_step()
    engine.close()
    
    print(f"ICs written to {saveDir} successfully as 7D variable")
    print(f"Shape: {global_shape}")
    return None



# -------- Parameter Reading and Initial Conditions Output -------- #

def load_params(filename_input):
    """ Load simulation parameters from the input file. """
    reader = ut.AthenaPKInputFileReader(filename_input)
    mesh_keys = ['nx1', 'nx2', 'nx3', 'x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max']
    mesh_params = {key: float(reader.get('parthenon/mesh', key)) for key in mesh_keys}
    
    problem_keys = ['rho_cloud_cgs', 'rho_wind_cgs', 'T_wind_cgs', 'r0_cgs', 'mach_shock']
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


def generate_ICs(params, rho_field, filename='ICs.bp', cloud_props=None):
    print("Generating ICs...")
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    mbar_over_kb = cloud_props.mbar/ut.constants.kb 

    full_box_rho = np.ones((nx3, nx2, nx1)) * params['rho_wind_cgs']
    start_idx = 40 + 3*8 #(nx2 - rho_field.shape[1])//2
    full_box_rho[:, start_idx:start_idx + rho_field.shape[1], :] = rho_field
    
    mom = np.zeros_like(full_box_rho)
    en = 0.5 * mom**2 / full_box_rho + np.ones_like(mom) * params['rho_wind_cgs'] * params['T_wind_cgs'] / (params['gamma'] - 1) /mbar_over_kb
    #mbar = cloud_props.mbar
    #p0 = calculate_pressure(params['T_wind_cgs'], params['rho_wind_cgs'], mbar)
    #Bx, By, Bz = gen_magnetic_field(params, p0, full_box_rho)
     # -- Apply inflow jump conditions to first 40 cells in y-direction --
    gamma = params['gamma']
    gm1 = gamma - 1
    rho_amb = params['rho_wind_cgs']
    T_amb = params['T_wind_cgs']
    mach = params['mach_shock']
    

    # Pressure and internal energy in ambient
    rhoe_amb = T_amb * rho_amb / (mbar_over_kb * gm1)
    pressure = gm1 * rhoe_amb

    # Rankine-Hugoniot jump conditions
    jump1 = (gamma + 1.0) / (gm1 + 2.0 / (mach * mach))
    jump2 = (2.0 * gamma * mach * mach - gm1) / (gamma + 1.0)
    jump3 = 2.0 * (1.0 - 1.0 / (mach * mach)) / (gamma + 1.0)

    rho_wind = rho_amb * jump1
    pressure_wind = pressure * jump2
    rhoe_wind = pressure_wind / gm1
    T_wind = pressure_wind / rho_wind * mbar_over_kb
    v_wind = jump3 * mach * np.sqrt(gamma * pressure / rho_amb)
    mom_wind = rho_wind * v_wind

    # Overwrite inflow region (first 40 cells in x2/y direction)
    inflow_slice = slice(0, 40)
    full_box_rho[:, inflow_slice, :] = rho_wind
    mom[:, inflow_slice, :] = mom_wind
    en[:, inflow_slice, :] = 0.5 * mom_wind**2 / rho_wind + rhoe_wind
    
    fields = (full_box_rho, mom, en)
    
    MeshBlockSize = (mbl3, mbl2, mbl1)
    MeshSize = (nx3, nx2, nx1)

    print('Begin adios generation')
    if filename.split(".")[-1] == "bin":
        ICs = gen_bin(fields, filename)
    elif filename.split(".")[-1] == "bp":
        ICs = gen_adios_streaming_new(MeshSize, MeshBlockSize, fields, filename)
        #ICs = gen_adios(MeshSize, MeshBlockSize, fields, filename)


    return ICs



# -------- Single Cloud Generation -------- #
def generate_sphere(filename_input, filename):
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
    
    
    save_path = os.path.join(localDir, filename)
    with open(save_path, "wb") as f:
        f.write(ICs.tobytes())

    print(f"Saved ICs {ICs.shape} to {save_path} ({os.path.getsize(save_path)} bytes).")
    return ICs

def generate_three_spheres(filename_input, separation_factor=100):
    params = load_params(filename_input)
    cloud_props = SingleCloudCC(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    x1min, x1max = params['x1min'], params['x1max']
    x2min, x2max = params['x2min'], params['x2max']
    x3min, x3max = params['x3min'], params['x3max']
    cell_size = (x2max-x2min)/nx2
    Rcloud = params['Rcloud']
    R = 0.5 * Rcloud
    print("this is cloud resolution:", R/cell_size/8)
    print("This will be the separation factor:", separation_factor)
    sep = separation_factor * R

    x1 = np.linspace(x1min, x1max, nx1)
    x2 = np.linspace(x2min, x2max, nx2)
    x3 = np.linspace(x3min, x3max, nx3)
    
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')

    # y-positions of the centers of the three spheres
    x2centre = (x2max + x2min) / 2
    y_start = x2min + cell_size* 60
    y_positions = [y_start, y_start + sep, y_start + 2*sep]
    print((np.array(y_positions) - y_start)/cell_size/8)

    # Compute distance from each of the three centers
    full_box_rho = np.full(X1.shape, params['rho_wind_cgs'])  # start with background

    for j, y_center in enumerate(y_positions):
        if y_center > x2max or y_center < x2min:
            raise ValueError(f"Sphere no. {j} at y_center={y_center} outside of bounds ({x2min}, {x2max})")
        distance = np.sqrt(X1**2 + (X2 - y_center)**2 + X3**2)
        inside_sphere = distance <= R
        full_box_rho[inside_sphere] = params['rho_cloud_cgs']
    

    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    mbar_over_kb = cloud_props.mbar/ut.constants.kb 
    
    mom = np.zeros_like(full_box_rho)
    en = 0.5 * mom**2 / full_box_rho + np.ones_like(mom) * params['rho_wind_cgs'] * params['T_wind_cgs'] / (params['gamma'] - 1) /mbar_over_kb
    
    fields = (full_box_rho, mom, en)
    
    MeshBlockSize = (mbl3, mbl2, mbl1)
    MeshSize = (nx3, nx2, nx1)

    ICs = gen_adios(MeshSize, MeshBlockSize, fields, 'ICs.bp')

    plt.imshow(ICs[0, :, :,nx3//2], cmap='viridis', norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.savefig("ICs_slice.png")
    plt.show()
    return ICs
    
def gen_magnetic_field(params, pressure, rho_field):
    """ Generate a magnetic field based on the density field. """
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    beta_in = params['beta_in']

    x = np.linspace(params['x1min'], params['x1max'], nx1)
    y = np.linspace(params['x2min'], params['x2max'], nx2)
    z = np.linspace(params['x3min'], params['x3max'], nx3)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    if beta_in is None: 
        return np.zeros((nx3, nx2, nx1)), np.zeros((nx3, nx2, nx1)), np.zeros((nx3, nx2, nx1))

    # Generate a tangled field with random vector potential
    np.random.seed(0)
    Ax = np.random.normal(0, 1, size=(nx3, nx2, nx1))
    Ay = np.random.normal(0, 1, size=(nx3, nx2, nx1))
    Az = np.random.normal(0, 1, size=(nx3, nx2, nx1))

    # Compute the magnetic field B = curl A
    def curl(Ax, Ay, Az, dx, dy, dz):
        Bx = (np.roll(Az, -1, axis=1) - np.roll(Az, 1, axis=1)) / (2*dy) - \
            (np.roll(Ay, -1, axis=2) - np.roll(Ay, 1, axis=2)) / (2*dz)
        By = (np.roll(Ax, -1, axis=2) - np.roll(Ax, 1, axis=2)) / (2*dz) - \
            (np.roll(Az, -1, axis=0) - np.roll(Az, 1, axis=0)) / (2*dx)
        Bz = (np.roll(Ay, -1, axis=0) - np.roll(Ay, 1, axis=0)) / (2*dx) - \
            (np.roll(Ax, -1, axis=1) - np.roll(Ax, 1, axis=1)) / (2*dy)
        return Bx, By, Bz

    dx = (params['x1max'] - params['x1min']) / nx1
    dy = (params['x2max'] - params['x2min']) / nx2  
    dz = (params['x3max'] - params['x3min']) / nx3
    Bx, By, Bz = curl(Ax, Ay, Az, dx, dy, dz)

    Brms2 = np.mean(Bx**2 + By**2 + Bz**2)
    B2_target = 2.0 * pressure / beta_in
    scale = np.sqrt(B2_target / Brms2)
    Bx *= scale
    By *= scale
    Bz *= scale

    mask = rho_field> 0
    Bx[~mask] = 0
    By[~mask] = 0
    Bz[~mask] = 0
        
    return Bx, By, Bz
# -------- Fractal ISM and Percolation Generation -------- #

def oldsimulate_percolation(dimensions, p, sigma):
    """ Simulate percolation with Gaussian smoothing. """
    print(f"Dimensions: {dimensions}, estimated memory per array: {np.prod(dimensions) * 8 / 1e9:.2f} GB")
    field = np.random.rand(*dimensions)
    smoothed_field = gaussian_filter(field, sigma=sigma) if not isinstance(sigma, tuple) else np.mean(
        [gaussian_filter(field, s) for s in np.linspace(*sigma, 5)], axis=0)
    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    print("Percolation field has been simulated.")
    return smoothed_field > threshold

def simulate_percolation(dimensions, p, sigma, chunk_size=4000):
    """ Simulate percolation with chunking to reduce memory. """
    nx1, nx2, nx3 = dimensions
    result = np.zeros(dimensions, dtype=bool)
    
    for i in range(0, nx2, chunk_size):
        end = min(i + chunk_size, nx2)
        chunk_dims = (nx1, end - i, nx3)
        
        field = np.random.rand(*chunk_dims)
        smoothed_field = gaussian_filter(field, sigma=sigma) if not isinstance(sigma, tuple) else np.mean(
            [gaussian_filter(field, s) for s in np.linspace(*sigma, 5)], axis=0)
        threshold = np.percentile(smoothed_field, 100 * (1 - p))
        result[:, i:end, :] = smoothed_field > threshold
        
    print("Percolation field has been simulated.")
    return result

def compute_p_values(sigmas, p_init):
    """ Compute p values for different sigmas. """
    sigmas_new = sorted(sigmas.copy())
    s0 = min(sigmas)
    p_values = []

    for sigma in sigmas_new:
        print("sigma:", sigma)
        p_values.append(p_init * (s0 / sigma) ** 3.)
    return p_values, sigmas_new


def create_ISM(filename_input='ism.in', ism_depth=1, fv=None, n_jobs=1):
    """ Generate ISM field with percolation and density fluctuations. """
    params = load_params(filename_input)
    cloud_props = SingleCloudCC(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
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
    if ',' not in kin:
        percolation_fields = Parallel(n_jobs=n_jobs)(delayed(simulate_percolation)(dimensions, fv, sigma) for _ in range(1))
        percolation_field = percolation_fields[0]
        
    if ',' in kin:
        sigmas = list(np.linspace(sigma[0], sigma[1], 40))
        p_values, sprime = compute_p_values(sigmas, fv)
        print("p_values:", p_values)
        print("sigmas:", sprime)

        buffers = Parallel(n_jobs=n_jobs)(delayed(simulate_percolation)(dimensions, p, s_sigma) for p,s_sigma in zip(p_values,sprime))
        percolation_field = np.sum(buffers, axis=0) > 0
        labeled_field, num_clumps = label(percolation_field)
        num_clumps_at_size = len(labeled_field[labeled_field > 8])
        print(f"Number of clumps: ", num_clumps_at_size)


    rho_field = np.where(percolation_field, params['rho_cloud_cgs'], params['rho_wind_cgs'])
    
    ics_output_file = str(params['reader'].get('job', 'bin_input_file'))

    ICs = generate_ICs(params, rho_field, filename=ics_output_file, cloud_props=cloud_props)
    print(f"ICs shape: {ICs.shape}")
    try:
        plt.imshow(ICs[0, :, :, nx3 // 2], cmap='viridis', norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.savefig("ICs_slice.png")
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}. Maybe you have selected the wrong data type for ICs?")

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
    #generate_three_spheres(filename_input=filename_input, separation_factor=args.r)

