import matplotlib.pyplot as plt
import numpy as np
from adioslib import read_adios
import os   
from adjust_ics import *
from rho_profile_y import compute_density_profile
from gen_strat import load_params

def rho_comparative(MeshSize, MeshBlockSize, nfields=1, localDir='.', gamma=5/3):
    # Read the arrays
    ICs = read_adios('ICs.bp', MeshSize, MeshBlockSize, nfields, localDir)
    reICs = read_adios('reICs.bp', MeshSize, MeshBlockSize, nfields, localDir)

    # Compute mean density and pressure profiles along y-axis (axis=2)
    y_coords, rho_ICs = compute_density_profile(ICs[0], axis=1)  # density
    _, rho_reICs = compute_density_profile(reICs[0], axis=1)

    
    pressure = (ICs[2] - 0.5 * ICs[0] * (ICs[1]/ICs[0])**2) * (gamma - 1)
    y_coords, p_ICs = compute_density_profile(pressure, axis=1)  # pressure (IEN)
    _, p_reICs = compute_density_profile(pressure, axis=1)

    # Compute normalized residuals
    rho_residual_norm = (rho_reICs - rho_ICs) / rho_ICs
    p_residual_norm = (p_reICs - p_ICs) / p_ICs

    # Plot two subplots
    fig, axes = plt.subplots(2, 2, figsize=(12,8), sharex=True)
    axes = axes.flatten()

    # 1. Density profile
    axes[0].plot(y_coords, rho_ICs, label='ICs.bp', lw=2, color='blue')
    axes[0].plot(y_coords, rho_reICs, label='reICs.bp', lw=2, color='cyan', linestyle='--')
    axes[0].set_ylabel('Density ⟨ρ⟩')
    axes[0].set_yscale('log')
    axes[0].set_title('Density Profile along Y')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Density normalized residual
    axes[1].scatter(y_coords, rho_residual_norm, color='red', label='Density residual / ICs')
    axes[1].set_ylabel('Normalized residual')
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_title('Density Normalized Residual along Y')

    # 3. Pressure profile
    axes[2].plot(y_coords, p_ICs, label='ICs.bp', lw=2, color='green')
    axes[2].plot(y_coords, p_reICs, label='reICs.bp', lw=2, color='lime', linestyle='--')
    axes[2].set_ylabel('Pressure ⟨P⟩')
    axes[2].set_yscale('log')
    axes[2].set_title('Pressure Profile along Y')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 4. Pressure normalized residual
    axes[3].scatter(y_coords, p_residual_norm, color='magenta', label='Pressure residual / ICs')
    axes[3].set_xlabel('Y index')
    axes[3].set_ylabel('Normalized residual')
    axes[3].grid(True)
    axes[3].legend()
    axes[3].set_title('Pressure Normalized Residual along Y')

    plt.tight_layout()
    plt.savefig('density_pressure_4panel.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'strat.in')

    params = load_params(filename_input)
    stratified_box = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    nfields = 3  # Adjust based on your data
    MeshSize = (nx1, nx2, nx3)  # Adjust based on your data
    MeshBlockSize = (mbl1, mbl2, mbl3)  # Adjust based on your data
    gamma = params['gamma']

    rho_comparative(MeshSize=MeshSize, MeshBlockSize=MeshBlockSize, nfields=nfields, localDir=localDir, gamma=gamma)