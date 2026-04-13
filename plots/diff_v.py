import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import re

from utils import *
from adjust_ics import *
from read_hdf5 import read_hdf5
from cooling import get_c_s, get_t_cool_n, get_t_cool_cgs

from stratified_box import StratifiedBox
from turbulent_box import TurbulentBox  # adjust imports
from t_grow_evol import mass_evolution, vel_evolution, compute_tgrow

# =========================
# --- MAIN ---
# =========================
if __name__ == "__main__":

    run_paths, saveFile = get_working_dirs()
    run_paths = list(run_paths)
    if 'r100' in run_paths[0]:
        run_paths.append('/viper/ptmp/ferhi/StratDisk/noturb/r100')

    mach_list = []
    last_velocities = []
    last_timescales = []

    for run in run_paths:
        print(f"\n--- {run} ---")
        try:
            try:
                sim = StratifiedBox(os.path.join(run, 'strat.in'), dir=run)
            except:
                sim = TurbulentBox(os.path.join(run, 'turbulence.in'), dir=run)

            code_time = sim.code_time_cgs
            code_length = sim.code_length_cgs

            t, m = mass_evolution(run)
            _, v = vel_evolution(run)

            if 'noturb' in run:
                Lambda_units = 1
            else:
                try:
                    slurm_file = os.path.join(run, 'slurm')
                    text = open(slurm_file).read()
                    lambda_units = re.search(r"cooling/lambda_units_cgs=([0-9.eE+-]+)", text)
                    Lambda_units = float(lambda_units.group(1))
                except:
                    Lambda_units = 1

        except Exception as e:
            print(f"Skipping {run}: {e}")
            continue

        # --- align time ---
        mask = 10**m > 1
        idx0 = np.argmax(mask) if np.any(mask) else 0
        t = t[idx0:] - t[idx0]
        m = m[idx0:] - m[idx0]
        v = v[idx0:]

        # --- physics ---
        t_myr, v_kms, tgrow_infall, tgrow_mix, tgrow_turb, tcc = \
            compute_tgrow(sim, t, v, m, code_time, code_length, Lambda_units)

        mass_linear = 10**m
        dmdt = np.gradient(mass_linear, t_myr)

        # --- last point calculations ---
        last_velocities.append(v_kms[-10])
        last_timescale = mass_linear[-10] / dmdt[-10] * u.Myr.to('s') * sim.g / 1e5
        print("This is last timescale:", last_timescale)
        last_timescales.append(last_timescale)

        if 'noturb' in run:
            mach = 0
        else:
            mach = float(sim.reader.get('problem/turbulence', 'Mach_drive'))
        mach_list.append(mach)

    # --- convert to numpy arrays ---
    mach_list = np.array(mach_list)
    last_velocities = np.array(last_velocities)
    last_timescales = np.array(last_timescales)

    # --- plot ---
    plt.figure(figsize=(8,6))
    plt.plot(mach_list, last_velocities, 'o', label='Last velocity (km/s)')
    plt.plot(mach_list, last_timescales, 's', label='Last mass timescale (converted)')
    plt.xlabel('Mach number')
    plt.ylabel('Value')
    plt.title('Last points as function of Mach number')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = f"/u/ferhi/Figures/{saveFile}_lastpoints_vs_mach.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.show()