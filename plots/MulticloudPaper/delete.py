import numpy as np
import h5py
from numba import njit
import yt
import dask.array as da
from joblib import Parallel, delayed

def generate_flat(vx_image, bins):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Convert to dask arrays for parallelism
    vx_image = da.from_array(vx_image, chunks=(vx_image.shape[0] // 4, vx_image.shape[1] // 4))
    
    flat_y, flat_z = np.meshgrid(bin_centers, bin_centers, indexing='ij')
    
    flat_y = flat_y.ravel()
    flat_z = flat_z.ravel()
    flat_vx = vx_image.ravel()
    
    mask = ~np.isnan(flat_vx)
    
    y_values = flat_y[mask]
    z_values = flat_z[mask]
    vx_values = flat_vx[mask]
    
    return y_values.compute(), z_values.compute(), vx_values.compute()



def generate_bins(dx,lmax,N_ij=5,length=128):
    
    '''
    At short radii, need to adapt the bin width to the pixel's distances
    Theoretically, the possible set of distances between neighboring pixels is given by:
    d_{i,j} = sqrt(i^2 + j^2) * pxl_size
    with i and j integers.
    
    # dx     : pixel size, kpc
    # lmax   : max length of the bins centers, kpc
    # N_ijk  : max integer value for i and j
    # length : total number of bins
    
    Returns:
    
    # log_centers: the logarithmic position of each bin centers (kpc)
    '''
    
    # Defining the list of sqrt(i^2 + j^2), will represent the center of bins.
    ij = []
    
    for i in range (N_ij):
        for j in range (N_ij):
                ij.append(np.sqrt(i**2 + j**2))
    
    ij = np.array(ij)*dx      # kpc
    ij = np.unique(ij)        # kpc
    ij = ij[1:6]              # kpc (truncation to the sixth element, can be modulated)
    
    # Compute the log10 of the bin centers
    log_centers   = np.log10(ij)
    
    # Compute the remaining of the list
    N_regular     = length - len(log_centers)
    start_regular = log_centers[-1] + (log_centers[-1] - log_centers[-2])
    end_regular   = np.log10(lmax)
    log_centers   = np.concatenate((log_centers,np.linspace(start_regular,end_regular,N_regular))) # Concatenate the adaptive bins list and the regular bins list
    
    return log_centers

def generate_bins_edges(bin_centers):
    """
    Generate histogram bin edges from bin centers.

    Parameters:
        bin_centers (numpy.ndarray): Array of bin center values.

    Returns:
        numpy.ndarray: Array of bin edge values.
    """
    
    # Calculate the differences between consecutive bin centers
    bin_widths = np.diff(bin_centers) / 2.0
    
    # Define the edges by extending outwards from the first and last bin centers
    bin_edges       = np.empty(len(bin_centers) + 1)
    bin_edges[1:-1] = bin_centers[:-1] + bin_widths
    bin_edges[0]    = bin_centers[0] - bin_widths[0]
    bin_edges[-1]   = bin_centers[-1] + bin_widths[-1]
    
    return bin_edges



def generate_VSF(y,z,v,rmax):
    
    
    if len(y) > 0: # if there at least one non-empty pixel
        
        Npxl = len(v)
        
        # Generate bin centers and edges
        log_bins_centers = generate_bins(dx, 2*rmax, N_ij=5, length=64) # Replace by relevant values
        log_bins_edges   = generate_bins_edges(log_bins_centers)
        bins_edges       = 10**log_bins_edges
        
        # Initialize storage for VSF and pair distribution
        vsf           = np.zeros(len(bins_edges) - 1) # Will be returned as the VSF value evaluated at the center of bins, so length is len(edges) - 1
        counts        = np.zeros(len(bins_edges) - 1) # Counter for the number of pairs falling into each bin
        
        @njit(parallel=True) # Uses numba acceleration
        def compute_vsf(y, z, v, bins_edges, vsf, counts):
            
            # Double for loop (the second starts to m + 1 to avoid counting pairs twice)
            for m in range(Npxl):
                for n in range(m+1,Npxl):
                    v1, v2 = v[m], v[n]
                    dv = np.abs(v1 - v2)  # By default, first-order line-of-sight VSF
                    dr = np.sqrt((y[m] - y[n])**2 + (z[m] - z[n])**2)
                    
                    # Find the corresponding bin index for dr
                    bin_idx = np.searchsorted(bins_edges, dr) - 1
                    
                    if 0 <= bin_idx < len(vsf):
                        vsf[bin_idx]    += dv      # Cumulative dv, will be normalized by counts once the double for loop is over
                        counts[bin_idx] += 1
        
        # Run the accelerated function
        compute_vsf(y, z, v, bins_edges, vsf, counts)
        
        # Normalize VSF, replace count by 1 for empty pixels (to avoid division by zero)
        vsf /= np.where(counts > 0, counts, 1)
        
        with h5py.File('VSF.h5', 'w') as f:
            
            # Store arrays as datasets in the HDF5 file
            f.create_dataset('vsf',  data=vsf)
            f.create_dataset('bins', data=10**log_bins_centers)
    return



if __name__ == "__main__":
    
    infile_name = '/raven/ptmp/ferhi/ISM_thinslab/Resolution_tests/fv01_7.6rcl_2L/out/parthenon.prim.00018.phdf'
    
    ds = yt.load(infile_name, default_species_fields="ionized")
    cg = ds.covering_grid(level=ds.max_level, left_edge = ds.domain_left_edge, dims = ds.domain_dimensions)

    #################################################

    ######## Extracting and writing the data ########

    #################################################

    # Conversion unit
    cm_to_kpc = 3.24078e-22

    nHp = cg['number_density'][:]
    ne  = cg['El_number_density'][:]
    T   = cg['temperature'][:]

    vx  = cg['velocity_x'][:]/1e5
    vy  = cg['velocity_y'][:]/1e5
    vz  = cg['velocity_z'][:]/1e5

    x   = cg['x'][:]*cm_to_kpc
    y   = cg['y'][:]*cm_to_kpc
    z   = cg['z'][:]*cm_to_kpc

    # Flattening data
    nHp = nHp.flatten()
    ne  = ne.flatten()
    T   = T.flatten()

    vx  = vx.flatten()
    vy  = vy.flatten()
    vz  = vz.flatten()

    x   = x.flatten()
    y   = y.flatten()
    z   = z.flatten()

    # Masking the temperature
    temperature_mask = T <= 3e4

    # Masking the data
    nHp = nHp[temperature_mask] # ionized hydrogen number density
    ne  = ne[temperature_mask]  # electron density number density
    T   = T[temperature_mask]

    vx = vx[temperature_mask]
    vy = vy[temperature_mask]
    vz = vz[temperature_mask]

    x  = x[temperature_mask]
    y  = y[temperature_mask]
    z  = z[temperature_mask]


    # Compute H-alpha emission, using whathever kind of weighting you want. 
    # Here is a simple density square weighting as an example
    # but you could also use more complicated emissivity's such as the one from:
    # "Hα and free-free emission from the warm ionized medium" - Dong and Draine 2011
    Halpha = ne * nHp

    # Define bin edges along y and z (assuming the box is centered at 0)
    # IMPORTANT: the binning here must be consistent with the grid structure!

    hw      = 18.6# Replace by the half-width of the cubic region loaded in step 1
    N_cells = 1172 # Replace by the number of cells along the y (or z) axis of the cubic region loaded in step 1
    bins    = np.linspace(-hw, hw, N_cells + 1) # Bins of the 2D histogram (the projection). N_cells + 1 as it corresponds to the bins edges!

    # Project along the line-of-sight
    hist_vx, _, _     = np.histogram2d(y, z, bins=(bins, bins), weights=vx * Halpha) # velocity_x * emission
    hist_Halpha, _, _ = np.histogram2d(y, z, bins=(bins, bins), weights=Halpha)      # emission only

    projected_vx = np.divide(hist_vx, hist_Halpha, where=hist_Halpha > 0)            # (velocity * emission) / emission (the actual weighted projection)

    # Set zero values to NaN (corresponding to pixels with no Halpha emission, empty pixels in other terms)
    projected_vx[hist_Halpha == 0] = np.nan
    print("Printing flat_y ...")
    flat_y,flat_z,flat_vx = generate_flat(projected_vx,bins)
    print("Generating VSF...")
    generate_VSF(flat_y, flat_z, flat_vx)