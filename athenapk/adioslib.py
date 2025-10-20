import os
import numpy as np
import adios2
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

def gen_bin(fields, filename, localDir='.'):
    
    print(len(fields))    
    ICs = np.stack(fields, axis=3).astype(np.float64)
    save_path = os.path.join(localDir, filename)
    
    with open(save_path, "wb") as f:
       f.write(ICs.tobytes())
    print(f"Saved ICs {ICs.shape} to {save_path} ({os.path.getsize(save_path)} bytes).")
 
    return ICs

def gen_adios(MeshSize, MeshBlockSize, fields, filename, localDir='.'):
    
    mbl3, mbl2, mbl1 = MeshBlockSize
    nx3, nx2, nx1 = MeshSize
    nz_blocks, ny_blocks, nx_blocks = (int(nx3/mbl3), int(nx2/mbl2), int(nx1/mbl1))
    x_indices, y_indices, z_indices = np.indices((nx_blocks, ny_blocks, nz_blocks))

    # Flatten the indices to get the logical locations for all blocks at once
    LogicalLocations = np.vstack((x_indices.ravel(), y_indices.ravel(), z_indices.ravel())).T
    n_blocks = LogicalLocations.shape[0]
    print(f"Generating {n_blocks} blocks of size {mbl3}x{mbl2}x{mbl1} for a total mesh size of {nx3}x{nx2}x{nx1}.")

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


def gen_adios_boundary(MeshSize, MeshBlockSize, fields_3d, n_ghosts, filename, boundary_face='x2_inner', localDir='.'):
    """
    Generate boundary condition data for a specific face.
    
    Args:
        MeshSize: (nx3, nx2, nx1) - full mesh size
        MeshBlockSize: (mbl3, mbl2, mbl1) - meshblock size
        fields_3d: list of 3D arrays for boundary (e.g., [rho_3d, mom_3d, en_3d])
                   Each should be shape (nx3, n_ghosts, nx1) for x2 boundaries
        n_ghosts: number of ghost cell layers
        filename: output filename
        boundary_face: which boundary ('x2_inner', 'x2_outer', etc.)
    """
    
    mbl3, mbl2, mbl1 = MeshBlockSize
    nx3, nx2, nx1 = MeshSize
    nz_blocks, ny_blocks, nx_blocks = (int(nx3/mbl3), int(nx2/mbl2), int(nx1/mbl1))
    
    # For x2 boundary, we have nz_blocks x nx_blocks
    x_indices, z_indices = np.indices((nx_blocks, nz_blocks))
    LogicalLocations = np.vstack((x_indices.ravel(), z_indices.ravel())).T
    n_blocks = LogicalLocations.shape[0]
    
    print(f"Generating {n_blocks} boundary blocks of size {mbl3}x{n_ghosts}x{mbl1} for boundary face.")
    
    # Pre-allocate block data for boundary with ghost cell dimension
    # Shape: (nz_blocks, ny_blocks, nx_blocks, n_fields, mbl3, n_ghosts, mbl1)
    block_data = np.zeros((nz_blocks, ny_blocks, nx_blocks, len(fields_3d), mbl3, n_ghosts, mbl1),
                          dtype=np.float64)
    
    # Reshape boundary fields to match meshblock structure
    meshblock_fields = []
    for field_3d in fields_3d:
        # field_3d is (nx3, n_ghosts, nx1), reshape to (nx_blocks, mbl3, n_ghosts, nz_blocks, mbl1)
        # First reshape to separate blocks in x and z dimensions
        reshaped = field_3d.reshape(nx_blocks, mbl3, n_ghosts, nz_blocks, mbl1)
        meshblock_fields.append(reshaped)
    
    # Fill block data
    for loc_x in range(nx_blocks):
        for loc_z in range(nz_blocks):
            for f in range(len(fields_3d)):
                # Assign to all y-locations (or just the boundary y-location)
                for loc_y in range(ny_blocks):
                    block_data[loc_z, loc_y, loc_x, f, :, :, :] = \
                        meshblock_fields[f][loc_x, :, :, loc_z, :]
    
    BCs = block_data
    
    saveDir = os.path.join(localDir, filename)
    shape = BCs.shape
    start = np.zeros_like(shape).tolist()
    count = BCs.shape
    nsteps = 1
    
    with Stream(saveDir, "w") as s:
        for _ in s.steps(nsteps):
            s.write("boundary", BCs, shape, start, count)
    
    print(f"Saved boundary array {BCs.shape} to {saveDir}. Size: {os.path.getsize(saveDir)} bytes.")
    
    return BCs


def read_adios(filename, MeshSize, MeshBlockSize, nfields, localDir='.'):
    """
    Read an ADIOS2 .bp file generated with gen_adios and reconstruct the full mesh arrays.
    
    Parameters
    ----------
    filename : str
        Name of the .bp file.
    MeshSize : tuple
        Global mesh size (nx3, nx2, nx1)
    MeshBlockSize : tuple
        Mesh block size (mbl3, mbl2, mbl1)
    nfields : int
        Number of fields.
    localDir : str
        Directory containing the .bp file.
    
    Returns
    -------
    global_array : np.ndarray
        Full mesh array of shape (nfields, nx3, nx2, nx1)
    """
    nx3, nx2, nx1 = MeshSize
    mbl3, mbl2, mbl1 = MeshBlockSize
    nz_blocks, ny_blocks, nx_blocks = int(nx3/mbl3), int(nx2/mbl2), int(nx1/mbl1)

    filepath = os.path.join(localDir, filename)
    print(f"Reading ADIOS2 file: {filepath}")

    with Stream(filepath, "r") as s:
        # Inspect available variables in first step
        for step in s.steps():  # usually one step for ICs
            print(f"Current step: {s.current_step()}")
            varnames = list(s.available_variables().keys())
            print("Available variables:", varnames)

            # Pick the first variable if nfields not known
            varname = varnames[0]
            print(f"Reading variable '{varname}'")

            # Read the full variable
            data = s.read(varname)  # numpy array

            # Stop after first step
            break

    blocks = np.array(data)  # shape: (nz_blocks, ny_blocks, nx_blocks, nfields, mbl3, mbl2, mbl1)
    global_array = np.zeros((nfields, nx3, nx2, nx1), dtype=blocks.dtype)

    # Reassemble global array from blocks
    for bz in range(nz_blocks):
        for by in range(ny_blocks):
            for bx in range(nx_blocks):
                for f in range(nfields):
                    z_start = bz * mbl3
                    y_start = by * mbl2
                    x_start = bx * mbl1
                    global_array[f,
                                 z_start:z_start+mbl3,
                                 y_start:y_start+mbl2,
                                 x_start:x_start+mbl1] = blocks[bz, by, bx, f, :, :, :]
    
    return global_array


def list_adios_vars(filename, localDir='.'):
    filepath = os.path.join(localDir, filename)
    print(f"Listing variables in {filepath}")

    adios = adios2.Adios()
    io = adios.declare_io("ReadIO")
    
    # Open the file for reading
    reader = io.open(filepath, adios2.Mode.Read)
    
    # Get all variable names
    varnames = io.AvailableVariables()
    
    print(f"Variables in file '{filename}':")
    for name in varnames:
        print(" -", name)
    
    reader.Close()
    return varnames