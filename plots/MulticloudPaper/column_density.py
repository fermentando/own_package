import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from adjust_ics import *
from utils import *
import constants as ct

def compute_column_density_and_fv(input_dir, threshold=4e-24):
    """
    Computes the column density and fv (number of clouds along y) for a 3D density field efficiently.
    
    Parameters:
    - density_field: 3D numpy array of shape (Nx, Ny, Nz)
    - threshold: Density threshold for clouds
    
    Returns:
    - column_density: 2D array of shape (Nx, Nz) with summed density over y
    - fv: 2D array of shape (Nx, Nz) with the number of distinct clouds along y
    - histograms of both as a function of x
    """
    
    sim = SingleCloudCC(os.path.abspath(os.path.join(input_dir, 'ism.in')), input_dir)
    params = sim.reader
    unit_length = float(params.get('units', 'code_length_cgs'))
    ICs, kval = sim._return_ICs()
    density_field = ICs[:,:,:,0]
    
    print("File loaded...")
    Nx, Ny, Nz = density_field.shape
    bin_size = 4
      
    Nx_binned = Nx // bin_size
    Nz_binned = Nz // bin_size

    # Initialize binned arrays
    column_density = np.zeros((Nx_binned, Nz_binned), dtype=np.float32)
    fv = np.zeros((Nx_binned, Nz_binned), dtype=int)

    # Create a boolean mask for thresholding
    cloud_mask = density_field > threshold

    # Loop over binned grid
    for bx in range(Nx_binned):
        for bz in range(Nz_binned):
            # Get the corresponding x, z indices for this bin
            x_start, x_end = bx * bin_size, (bx + 1) * bin_size
            z_start, z_end = bz * bin_size, (bz + 1) * bin_size
            
            # Sum density over y for this bin (column density)
            column_density[bx, bz] = np.sum(density_field[x_start:x_end, :, z_start:z_end])
            
            # Count clumps in the bin
            cloud_section = cloud_mask[x_start:x_end, :, z_start:z_end]
            labeled_y, num_clumps = label(cloud_section)
            fv[bx, bz] = num_clumps
    
    
    print(fv)
    print("Cloud counts...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Histogram of column density as a function of binned x
    axes[0].bar(np.arange(Nx_binned), np.sum(column_density, axis=1)/(ct.mu * ct.CONST_amu)/np.sum(np.ones_like(column_density)* unit_length, axis=1), color='b', alpha=0.6)
    axes[0].set_xlabel("Binned x")
    axes[0].set_ylabel("Total Column Density")
    axes[0].set_title("Histogram of Column Density vs Binned x")

    # Histogram of fv as a function of binned x
    axes[1].bar(np.arange(Nx_binned), np.sum(fv, axis=1), color='r', alpha=0.6)
    axes[1].set_xlabel("Binned x")
    axes[1].set_ylabel("Total fv (Number of Dense Clumps)")
    axes[1].set_title("Histogram of fv vs Binned x")
    plt.tight_layout()
    plt.show()
    
    return column_density, fv


if __name__ == "__main__":
    input_dir = "/ptmp/ferhi/ISM_thinslab/kc/fv01_3.8rcl"
    
    compute_column_density_and_fv(input_dir, threshold = 4e-24)
    