import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import re

from utils import *
from adjust_ics import *
from read_hdf5 import read_hdf5
from cooling import get_c_s, get_t_cool_n, get_t_cool_cgs


# =========================
# --- DATA ---
# =========================

def read_hst(run):
    data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
    return np.where(data == 0, 1e-22, data)


def mass_evolution(run):
    data = read_hst(run)
    t = data[:, 0]
    m = np.log10(data[:, 10] / data[0, 10])
    return t, m


def vel_evolution(run):
    data = read_hst(run)
    t = data[:, 0]
    v = -(data[:, 13] / data[:, 10])
    return t, v

def turb_speeds(run):
    data = read_hst(run)
    t = data[:, 0]
    v1 = (data[:, 12] / data[:, 10])
    v3 = (data[:, 14] / data[:, 10])

    v = np.sqrt(v1**2 + v3**2) 
    return t, v


# =========================
# --- PHYSICS ---
# =========================

def profile_n(r, sim):
    n0, a, H = sim.rho_base/sim.mbar, sim.a_over_H, sim.H/u.kpc.to('cm')
    print(f'These are n0: {n0}, a: {a}, H: {H}')
    return n0 * np.exp(-a * ((1 + (r / (a * H))**2)**0.5 - 1))


def compute_tgrow(sim, t, v, m, code_time, code_length, Lambda_units):
    """All tgrow-related quantities"""

    f_A = 0.23

    t_myr = t * code_time / u.Myr.to('s')
    v_kms = v * code_length / code_time / 1e5
    mach = float(sim.reader.get('problem/turbulence', 'Mach_drive'))
    if 'noturb' in sim.dir:
        print('This is noturb, setting mach to 0')
        mach = 0

    r = np.where(
    v > 0,
    sim.y_centre / 1e3 - v * t / 1e3,
    sim.y_centre / 1e3
)
    mrat = 10**m

    tcool0 = get_t_cool_cgs(sim.cloud_rho, sim.T_cloud, sim.mbar) * Lambda_units / u.Myr.to('s')
    tgrow0 = 100* (f_A / 0.23)  * (sim.chi/100.) * (sim.r_cloud_inserted / 100.) *\
            (sim.r_cloud_inserted / 100.)**(-0.25) * (tcool0 / 0.03 )**(0.25)* ( 1 + mach)**(2/3)

    w_KH = np.minimum(1, 3 *  np.sqrt(sim.chi) *
        (sim.r_cloud_inserted / 1000) /
        r)


    tcool_cl = get_t_cool_n( sim.T_cloud, profile_n(r, sim) * sim.chi, sim.mbar) * Lambda_units / u.Myr.to('s')
    cs = get_c_s(sim.T_base) / 1e5
    vrat = 150. / np.minimum(abs(v_kms), cs) 
    vs = mach * cs * 1e5
    tcoolrat = tcool_cl / tcool0
    print("This is initial mrat: ", mrat[0])
    rhorat = profile_n(r, sim) / profile_n(sim.y_centre / 1e3, sim)

    tgrow_infall =  w_KH * tgrow0 * u.Myr.to('s') * vrat**(3/5.) * tcoolrat**(1/4.) * (mrat / rhorat)**(1-5/6.)


    tgrow_mix = sim.chi / 15e5 * sim.r_cloud_inserted * sim.code_length_cgs
    tgrow_turb = sim.chi * np.sqrt(get_t_cool_cgs(sim.cloud_rho, sim.T_cloud, sim.mbar) * Lambda_units  * sim.r_cloud_inserted * sim.code_length_cgs / (mach * cs* 1e5))

    tcc = (
        sim.r_cloud_inserted * sim.code_length_cgs * sim.chi**0.5 /
        np.maximum(v * code_length / code_time, 1e-10)
    )

    return t_myr, v_kms, tgrow_infall, tgrow_mix, tgrow_turb, tcc 



# =========================
# --- MAIN ---
# =========================

if __name__ == "__main__":

    plt.style.use('custom_plot')

    # 2x2 layout
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex=True
    )

    ax_vel_infall = axes[0, 0]
    ax_vel_turb   = axes[0, 1]
    ax_mdot_infall = axes[1, 0]
    ax_mdot_turb   = axes[1, 1]

    run_paths, saveFile = get_working_dirs()
    run_paths = list(run_paths)
    if 'r100' in run_paths[0]:  # if the first path is the noturb one, we can skip it since it's already in the list
        run_paths.append('/viper/ptmp/ferhi/StratDisk/noturb/r100')

    for j, run in enumerate(run_paths):

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
                print('Lambda units =1 ')
            else:
                
                try:
                    slurm_file = os.path.join(run, 'slurm')

                    text = open(slurm_file).read()

                    tlim = re.search(r"parthenon/time/tlim=([0-9.eE+-]+)", text)
                    lambda_units = re.search(r"cooling/lambda_units_cgs=([0-9.eE+-]+)", text)
                    Lambda_units = float(lambda_units.group(1))
                    print('these are the Lambda units: ', Lambda_units)
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

        label = os.path.basename(run)

        # --- physics ---
        t_myr, v_kms, tgrow_infall, tgrow_mix, tgrow_turb, tcc = \
            compute_tgrow(sim, t, v, m, code_time, code_length, Lambda_units)

        mass_linear = 10**m
        dmdt = mass_linear * np.log(10) * np.gradient(m, t_myr)  # convert from log to linear and then take gradient
        window = int(50)  # choose something reasonable

        kernel = np.ones(window) / window
        dmdt_smooth = np.convolve(dmdt, kernel, mode='same')

        # =========================
        # --- PLOTS ---
        # =========================

        # --- velocity row ---
        line, = ax_vel_infall.plot(t_myr, v_kms, label=label, alpha=0.8)
        ax_vel_infall.plot(
            t_myr,
            mass_linear / dmdt_smooth * u.Myr.to('s') * sim.g / 1e5,
            linestyle='-.',
            color=line.get_color(),
            alpha=0.8
        )

        line, = ax_vel_turb.plot(t_myr, v_kms, label=label, alpha=0.8)
        #print('This is g t_grow_turb: ', tgrow_turb[-1] * sim.g / 1e5)
        ax_vel_turb.axhline(
            tgrow_turb * sim.g / 1e5,
            linestyle='--',
            color=line.get_color(),
            alpha=0.8
        )

        ax_vel_turb.axhline(
            tgrow_mix * sim.g / 1e5,
            linestyle=':',
            color=line.get_color(),
            alpha=0.8
        )

        # --- mdot row ---
        ratio = mass_linear / dmdt_smooth

        ax_mdot_infall.plot(
            t_myr,
            ratio / (tgrow_infall / u.Myr.to('s')),
            linestyle='-.',
            alpha=0.8
        )

        ax_mdot_turb.plot(
            t_myr,
            ratio / (tgrow_turb / u.Myr.to('s')),
            linestyle='--',
            alpha=0.8
        )

    # =========================
    # --- FORMAT ---
    # =========================

    # Titles
    ax_vel_infall.set_title('Velocity vs tgrow_infall')
    ax_vel_turb.set_title('Velocity vs tgrow_turb')

    ax_mdot_infall.set_title(r'$m/\dot{m}$ vs tgrow_infall')
    ax_mdot_turb.set_title(r'$m/\dot{m}$ vs tgrow_turb')

    # Labels
    ax_vel_infall.set_ylabel('Velocity (km/s)')
    ax_mdot_infall.set_ylabel(r'$m / (\dot{m}  tgrow_{infall})$')
    ax_mdot_turb.set_ylabel(r'$m / (\dot{m}  tgrow_{turb})$')

    ax_mdot_infall.set_xlabel('Time (Myr)')
    ax_mdot_turb.set_xlabel('Time (Myr)')

    # Limits (optional reuse)
    ax_vel_infall.set_ylim(-40, 100)
    ax_vel_turb.set_ylim(-40, 100)

    ax_mdot_infall.set_ylim(1e-2, 10)
    ax_mdot_infall.set_yscale('log')
    ax_mdot_turb.set_ylim(1e-2, 10)
    ax_mdot_turb.set_yscale('log')

    # Legends (only left column to avoid clutter)
    ax_vel_infall.legend()
    ax_mdot_infall.legend()

    plt.tight_layout()

    save_path = f"/u/ferhi/Figures/{saveFile}_2x2.png"
    print(f"Saved to: {save_path}")
    plt.savefig(save_path)