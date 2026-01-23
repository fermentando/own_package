import os
import glob
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt
import astropy.units as u
import yt


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

def yt_coldgs(run, output_dir=None):
        ds = yt.load(os.path.join(run))
        temp = ds.all_data()[('gas', 'temperature')] 
        mass = ds.all_data()[('gas', 'mass')]
        coldg = np.sum(mass[temp <= 2e4])
        ts = ds.current_time
        
        return ts, coldg

def vel_evolution(run):
    data = read_hst(run)
    vel_ind = 13
    mass_ind = 10
    timeseries = data[:, 0]
    velocity = -(data[:, vel_ind] / data[:, mass_ind])
    return timeseries, velocity

if __name__ == "__main__":
    plt.style.use('custom_plot')
    plot_yt = False
    plot_hst = True



    
    run_paths, saveFile = get_working_dirs()

    if plot_hst:
            # Two subplots: mass evolution (top), velocity (bottom)
        fig, (ax_mass, ax_vel) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        for j, run in enumerate(run_paths):
            sim = TurbulentBox(os.path.join(run, 'turbulence.in'), dir=run)
            code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
            code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
            files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))

            try:
                times, norm_mass, cgout, wgout, total = mass_evolution(run, gout=True)
                tvs, vel = vel_evolution(run)
            except Exception as e:
                print(f"Skipping {run}: {e}")
                continue

            # Identify blob injection time
            print(norm_mass)
            try:
                mask = 10**norm_mass > 1
                idx0 = np.where(mask)[0][0]
            except:
                mask = 10**norm_mass > 0   
                idx0 = np.where(mask)[0][0]

            timeseries = times[idx0:] - times[idx0]
            norm_mass = norm_mass[idx0:]- norm_mass[idx0]
            vel = vel[idx0:] 


            label = run.split('/')[-1]
            plt.style.use('custom_plot')

            # --- MASS EVOLUTION subplot ---
            ax_mass.plot(timeseries * sim.t_eddy * code_time_cgs / u.Myr.to('s'), norm_mass, label=f"{label} mass", alpha=0.8)
            # if np.sum(cgout) > 10 * len(cgout) * 1e-22:
            #     ax_mass.plot(timeseries, cgout, alpha=0.5, label=f"{label} cgout")
            # if np.sum(wgout) > 10 * len(wgout) * 1e-22:
            #     ax_mass.plot(timeseries, wgout, alpha=0.3, label=f"{label} wgout")
            # if (np.sum(cgout) > 10 * len(cgout) * 1e-22) and (np.sum(wgout) > 10 * len(wgout) * 1e-22):
            #     ax_mass.plot(timeseries, total, color='black', linestyle='--', alpha=0.3, label="total")

            # --- VELOCITY EVOLUTION subplot ---
            ax_vel.plot(timeseries * sim.t_eddy * code_time_cgs / u.Myr.to('s'), vel * code_length_cgs / code_time_cgs /1e5, label=f"{label} velocity", alpha=0.8)

        # --- LABELS and formatting ---
        ax_mass.set_ylabel(r'$\log(m/m_0)$')
        ax_mass.set_ylim(bottom=-2)
        ax_mass.set_xlim(left=0)
        ax_mass.legend()

        ax_vel.set_xlabel(r't [Myr]')
        ax_vel.set_ylabel(r'infall speed $(km/s)$')
        ax_vel.set_xlim(left=0)
        ax_vel.legend()


            #if "fv01_narrow" in run:
    if plot_yt:
        for j, pathrun in enumerate(run_paths):
            runs = glob.glob(os.path.join(pathrun, 'out/parthenon.prim.[0-9]*.phdf'))[6:]
            print(len(runs))
            initial_mass = None
            for run in runs:
                ts, coldg = yt_coldgs(run)
            #ts, coldg = run_parallel(runs, func=yt_coldgs_hdf, num_workers=N_procs, output_dir=None)
                label = None
                if (initial_mass == None) and (coldg > 0): initial_mass = coldg
                label = run.split('/')[-1] + (' Hst' if 'Hst' in run else '')
                plt.scatter(ts, np.log10(coldg/initial_mass), label=label, color='blue')
                print(f"Cold gas mass: {np.log10(coldg/initial_mass)}")


    plt.tight_layout()
    print(f"Saved to: /u/ferhi/Figures/{saveFile}_mass_vel_evolution.png")
    plt.savefig(f"/u/ferhi/Figures/{saveFile}_mass_vel_evolution.png")
    

