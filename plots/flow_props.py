import os
import glob
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt

plt.style.use('custom_plot')

def read_hst(run): 
    data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
    data = np.where(data == 0, 1e-22, data)
    return data

def mass_evolution(run, gout=False):
    data = read_hst(run)
    mass_ind = 10
    norm_mass = np.log10(data[:, mass_ind] / data[0, mass_ind])
    timeseries = data[:, 0]
    
    wgout = np.zeros_like(timeseries)
    cgout = np.zeros_like(timeseries)
    total = norm_mass
    
    if gout: 
        wgout = np.log10(data[:, -2] / data[0, mass_ind]) 
        cgout = np.log10(data[:, -3] / data[0, mass_ind])
        total = np.log10((data[:, mass_ind] + data[:, -2] + data[:, -3]) / data[0, mass_ind])
    return timeseries, norm_mass, cgout, wgout, total

def vel_evolution(run):
    data = read_hst(run)
    vel_ind = 13
    mass_ind = 10
    timeseries = data[:, 0]
    velocity = abs(data[:, vel_ind] / data[:, mass_ind])
    return timeseries, velocity

if __name__ == "__main__":
    plot_yt = False
    plot_hst = True
    problem_name = 'stratified_box'

    # Two subplots: mass evolution (top), velocity (bottom)
    fig, (ax_mass, ax_vel) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    run_paths, saveFile = get_working_dirs()

    for j, run in enumerate(run_paths):
        sim = StratifiedBox(os.path.join(run, 'restrat.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))

        try:
            timeseries, norm_mass, cgout, wgout, total = mass_evolution(run, gout=True)
            tvs, vel = vel_evolution(run)
        except Exception as e:
            print(f"Skipping {run}: {e}")
            continue

        mask = ~np.isnan(norm_mass)
        timeseries = timeseries[mask]
        norm_mass = norm_mass[mask]

        label = run.split('/')[-1]
        plt.style.use('custom_plot')

        # --- MASS EVOLUTION subplot ---
        ax_mass.plot(timeseries, norm_mass, label=f"{label} mass", alpha=0.8)
        if np.sum(cgout) > 10 * len(cgout) * 1e-22:
            ax_mass.plot(timeseries, cgout, alpha=0.5, label=f"{label} cgout")
        if np.sum(wgout) > 10 * len(wgout) * 1e-22:
            ax_mass.plot(timeseries, wgout, alpha=0.3, label=f"{label} wgout")
        if (np.sum(cgout) > 10 * len(cgout) * 1e-22) and (np.sum(wgout) > 10 * len(wgout) * 1e-22):
            ax_mass.plot(timeseries, total, color='black', linestyle='--', alpha=0.3, label="total")

        # --- VELOCITY EVOLUTION subplot ---
        ax_vel.plot(timeseries, vel * code_length_cgs / code_time_cgs /1e5, label=f"{label} velocity", alpha=0.8)

    # --- LABELS and formatting ---
    ax_mass.set_ylabel(r'$\log(m/m_0)$')
    ax_mass.set_ylim(bottom=-2)
    ax_mass.legend()

    ax_vel.set_xlabel(r't [code units]')
    ax_vel.set_ylabel(r'$|v| / m$')
    ax_vel.legend()

    plt.tight_layout()
    print(f"Saved to: /u/ferhi/Figures/{saveFile}_mass_vel_evolution.png")
    plt.savefig(f"/u/ferhi/Figures/{saveFile}_mass_vel_evolution.png")
    

