import numpy as np
import h5py
import matplotlib.pyplot as plt
from read_hdf5 import read_hdf5
from adjust_ics import *
import os
from gen_strat import gen_rho_strat
from scipy.optimize import curve_fit

def compute_density_profile(rho, axis=1):
    """
    Compute mean density profile along given axis (default: y-axis).
    Returns mean rho and coordinate array (assuming unit spacing).
    """
    # Take mean over all other axes
    profile = np.mean(rho, axis=tuple(i for i in range(rho.ndim) if i != axis))
    y = np.arange(profile.size)
    return y, profile

def hydrostatic_profile(Y, rho0, a, H):
    """Analytic model: rho = rho0 * exp(-a*(sqrt(1 + (Y/(a*H))**2) - 1))"""
    return rho0 * np.exp(-a * (np.sqrt(1 + (Y / (a * H))**2) - 1))

def main():
    # --- Input files ---
    file_main = "/home/fernando/Runs/StratSimple/parthenon.prim.00012.phdf"


    # --- Read rho field ---
    read_file = read_hdf5(file_main, ['rho', 'prs'])
    rho_main = read_file['rho']
    prs_main = read_file['prs']
    print(f"rho_main shape: {rho_main.shape}")


    # Read init data
    strat_sim = StratifiedBox(os.path.abspath(os.path.join(file_main, '../strat.in')), '.')
    ymin, ymax = float(strat_sim.reader.get('parthenon/mesh', 'x2min')), float(strat_sim.reader.get('parthenon/mesh', 'x2max'))
    ny = rho_main.shape[1]
    print(f"ymin: {ymin}, ymax: {ymax}, ny: {ny}")
    dy = (ymax - ymin) / ny

    # --- Compute profiles along y ---
    y_main, prof_main = compute_density_profile(rho_main, axis=1)
    y_main = y_main * dy + ymin# Convert to physical units

    prs_y_main, prs_prof_main = compute_density_profile(prs_main, axis=1)
    prs_y_main = prs_y_main * dy + ymin# Convert to physical units

    # --- Mirror profiles ---
    rho_ref, _ = gen_rho_strat(os.path.abspath(os.path.join(file_main, '../strat.in')))
    y_ref, prof_ref = compute_density_profile(rho_ref, axis=1)
    y_ref = y_ref[4:-4] * dy + ymin # Convert to physical units
    prof_ref = prof_ref[4:-4]  # Remove boundaries

    strat_run = StratifiedBox(os.path.abspath(os.path.join(file_main, '../strat.in')), '.')
    prs_ref = rho_ref * ut.constants.kb * 1e6 / strat_run.mbar
    prs_y_ref, prs_prof_ref = compute_density_profile(prs_ref, axis=1)
    prs_y_ref = prs_y_ref[4:-4] * dy + ymin # Convert to physical units
    prs_prof_ref = prs_prof_ref[4:-4]  # Remove boundaries


    # Use only the positive half to fit (symmetry)
    mask = y_main >= 0
    y_fit = y_main[mask]
    rho_fit = prof_main[mask]

    # Initial guesses: rho0, a, H
    p0 = [1e-22, 0.1, 500]

    popt, pcov = curve_fit(hydrostatic_profile, y_fit, rho_fit, p0=p0, maxfev=10000)
    rho0_fit, a_fit, H_fit = popt
    print(f"Best fit parameters:")
    print(f"  rho0 = {rho0_fit:.3e}")
    print(f"  a    = {a_fit:.3f}")
    print(f"  H    = {H_fit:.3f}")

    rho_bestfit = hydrostatic_profile(y_main, *popt)





    # --- Plot with two subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Density subplot
    ax1.plot(y_ref, prof_ref, label="Reference", lw=2, color='gray', linestyle='--')
    ax1.plot(y_main, prof_main, label="Current", lw=2, color='C0')
    #ax1.plot(y_main, rho_bestfit, label="Best-fit model", lw=2, color='C3', linestyle='-.')
    ax1.set_xlabel("y (grid units)")
    ax1.set_ylabel("Average Density ⟨ρ⟩")
    ax1.set_title("Density Profile along Y")
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Pressure subplot
    ax2.plot(prs_y_ref, prs_prof_ref, label="Reference", lw=2, color='gray', linestyle='--')
    ax2.plot(prs_y_main, prs_prof_main, label="Current", lw=2, color='C1')
    #ax2.plot(prs_y_main, prs_bestfit, label="Best-fit model", lw=2, color='C4', linestyle='-.')
    ax2.set_xlabel("y (grid units)")
    ax2.set_ylabel("Average Pressure ⟨P⟩")
    ax2.set_title("Pressure Profile along Y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')    

    plt.tight_layout()
    plt.savefig("profiles_y.png", dpi=300)

if __name__ == "__main__":
    main()
