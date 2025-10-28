import numpy as np
from read_hdf5 import read_hdf5 
import matplotlib.pyplot as plt
import os
from adjust_ics import *
from gen_strat import load_params, insert_sphere
from adioslib import *
import matplotlib

def rescale_to_rms_mach(file, target_rms_Ms, filename_input='restart.in', localDir='.', 
                        filepath=None):
    """
    Rescale the temperature/pressure in a simulation to achieve a target RMS Mach number.
    
    Parameters:
    -----------
    file : str
        Path to the HDF5 file containing the data
    target_rms_Ms : float
        Target RMS Mach number to rescale to
    filename_input : str
        Input parameter file (default: 'restart.in')
    localDir : str
        Local directory for output (default: '.')
    filepath : str, optional
        Alternative filepath for reading data
    
    Returns:
    --------
    rescaled_data : ndarray
        Rescaled data array ready for ADIOS2 output
    """
    
    
    # Read the data
    print("Reading HDF5 file...")
    data = read_hdf5(file, ['rho', 'prs', 'vel1', 'vel2', 'vel3'], filepath=filepath)
    
    full_rho = data['rho']
    full_prs = data['prs']
    full_vel1 = data['vel1']
    full_vel2 = data['vel2']
    full_vel3 = data['vel3']
    
    # Load parameters
    params = load_params(filename_input)
    gamma = params['gamma']
    
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) 
                        for i in range(1, 4))
    
    # Get domain size
    code_length_cgs = float(params['reader'].get('units', 'code_length_cgs'))
    Lx = (params['x1max'] - params['x1min']) * code_length_cgs
    Ly = (params['x2max'] - params['x2min']) * code_length_cgs
    Lz = (params['x3max'] - params['x3min']) * code_length_cgs
    volume = Lx * Ly * Lz
    
    print(f"Domain size: Lx={Lx:.3e}, Ly={Ly:.3e}, Lz={Lz:.3e}")
    print(f"Total volume: {volume:.3e}")
    
    # Calculate cell volumes (assuming uniform grid for simplicity)
    # If you have non-uniform grid, you'll need to calculate cell volumes properly
    dx = Lx / nx1
    dy = Ly / nx2
    dz = Lz / nx3
    cell_volume = dx * dy * dz
    
    # Calculate kinetic energy density
    kin_en_density = 0.5 * full_rho * (full_vel1**2 + full_vel2**2 + full_vel3**2)
    
    # Calculate local Mach number squared: Ms^2 = 2 * KE / (gamma * P)
    # This is equivalent to Ms^2 = v^2 / (gamma * P / rho) = v^2 / c_s^2
    local_Ms2 = 2.0 * kin_en_density / (gamma * full_prs)
    
    # Calculate RMS Mach number squared (volume-weighted average)
    Ms2_sum = np.sum(local_Ms2 * cell_volume)
    current_rms_Ms = np.sqrt(Ms2_sum / volume)
    
    print(f"\nCurrent RMS Mach number: {current_rms_Ms:.4f}")
    print(f"Target RMS Mach number: {target_rms_Ms:.4f}")
    
    # Check for valid target
    if target_rms_Ms <= 0:
        raise ValueError("Target Mach number must be positive!")
    
    # Calculate rescaling factor
    # We want: Ms_new^2 = Ms_target^2
    # Since Ms^2 ∝ KE/P, and we keep KE constant, we need P_new = P_old / norm
    # where norm = (Ms_target / Ms_current)^2
    norm = (target_rms_Ms / current_rms_Ms)**2
    
    print(f"Rescaling factor: {norm:.6f}")
    
    # Rescale pressure (or equivalently, temperature)
    # This is done by changing the internal energy while keeping kinetic energy constant
    full_prs_rescaled = full_prs * norm
    
    # Calculate new total energy
    # E_total = E_internal + E_kinetic
    # E_internal = P / (gamma - 1)
    en_rescaled = full_prs_rescaled / (gamma - 1.0) + kin_en_density
    
    # Verify the rescaling
    local_Ms2_new = 2.0 * kin_en_density / (gamma * full_prs_rescaled)
    Ms2_sum_new = np.sum(local_Ms2_new * cell_volume)
    new_rms_Ms = np.sqrt(Ms2_sum_new / volume)
    
    print(f"Achieved RMS Mach number: {new_rms_Ms:.4f}")
    print(f"Relative error: {abs(new_rms_Ms - target_rms_Ms) / target_rms_Ms * 100:.2e}%")
    
    # Calculate momentum (keep velocities unchanged)
    mom1 = full_rho * full_vel1
    mom2 = full_rho * full_vel2
    mom3 = full_rho * full_vel3
    
    # Optional: insert sphere if needed (adjust parameters as in your original code)
    if 'r_cloud_inserted' in params and params['r_cloud_inserted'] > 0:
        print("\nInserting sphere...")
        stratified_box = StratifiedBox(filename_input, 
                                       os.path.abspath(os.path.join(filename_input, '..')))
        mbar_over_kb = stratified_box.mbar / ut.constants.kb
        
        # For simplicity, using total momentum magnitude
        mom_total = np.sqrt(mom1**2 + mom2**2 + mom3**2)
        
        rho_cloud, mom_cloud, en_cloud = insert_sphere(
            full_rho, mom_total, en_rescaled,
            r=params['r_cloud_inserted'] * code_length_cgs,
            T_cloud=params['T_cloud'],
            mbar_over_kb=mbar_over_kb,
            gamma=gamma,
            T_base=params['T_base'],
            x_range=(params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
            y_range=(params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
            z_range=(params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs),
            inplace=False
        )
        
        fields = (rho_cloud, mom_cloud, en_cloud)
    else:
        fields = (full_rho, mom1, mom2, mom3, en_rescaled)
    
    # Generate ADIOS2 file
    print("\nGenerating ADIOS2 file...")
    mesh_shape = (nx1, nx2, nx3)
    mb_shape = (mbl1, mbl2, mbl3)
    
    output_filename = 'rescaled_ICs.bp'
    rescaled_data = gen_adios(mesh_shape, mb_shape, fields, output_filename, localDir)
    
    print(f"Output shape: {rescaled_data.shape}")
    
    # Plot a slice for verification
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(full_rho[:, :, nx3 // 2], cmap='viridis', 
                  norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Density')
        plt.title('Density (slice)')
        
        plt.subplot(1, 3, 2)
        plt.imshow(full_prs_rescaled[:, :, nx3 // 2], cmap='hot', 
                  norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Pressure')
        plt.title('Rescaled Pressure (slice)')
        
        plt.subplot(1, 3, 3)
        plt.imshow(local_Ms2_new[:, :, nx3 // 2], cmap='plasma')
        plt.colorbar(label='$M_s^2$')
        plt.title('Local Mach Number Squared (slice)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(localDir, "rescaled_slice.png"), dpi=150)
        plt.show()
        print(f"Saved visualization to {os.path.join(localDir, 'rescaled_slice.png')}")
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    return rescaled_data


# Example usage:
if __name__ == "__main__":
    # Adjust these parameters for your use case
    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'strat.in')
    sim = StratifiedBox(filename_input, localDir)
    target_mach = sim.Mach  # Target RMS Mach number
    
    rescaled_data = rescale_to_rms_mach(
        file=input_file,
        target_rms_Ms=target_mach,
        filename_input='restart.in',
        localDir='.'
    )
    
    print("\nRescaling complete!")