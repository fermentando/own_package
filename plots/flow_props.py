import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

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


# =========================
# --- PHYSICS ---
# =========================

def profile_n(r, sim):
    n0, a, H = sim.rho_base/sim.mbar, sim.a_over_H, sim.H/u.kpc.to('cm')
    return n0 * np.exp(-a * ((1 + (r / (a * H))**2)**0.5 - 1))


def compute_tgrow(sim, t, v, m, code_time, code_length, Lambda_units):
    """All tgrow-related quantities"""

    f_A = 0.23

    t_myr = t * code_time / u.Myr.to('s')
    v_kms = v * code_length / code_time / 1e5
    mach = float(sim.reader.get('problem/turbulence', 'Mach_drive'))

    r = np.where(
    v > 0,
    sim.y_centre / 1e3 - v * t / 1e3,
    sim.y_centre / 1e3
)
    mrat = 10**m

    tcool0 = get_t_cool_cgs(sim.cloud_rho, sim.T_cloud, sim.mbar) * Lambda_units / u.Myr.to('s')
    tgrow0 = 100* (f_A / 0.23)  * (sim.chi/100.) * (sim.r_cloud_inserted / 100.) *\
            (sim.r_cloud_inserted / 100.)**(-0.25) * (tcool0 / 0.03 )**(0.25) * ( 1 + mach)**(2/3)

    w_KH = np.minimum(1, 3 *  np.sqrt(sim.chi) *
        (sim.r_cloud_inserted / 1000) /
        r)


    tcool_cl = get_t_cool_n( sim.T_cloud, profile_n(r, sim) * sim.chi, sim.mbar) * Lambda_units / u.Myr.to('s')
    cs = get_c_s(sim.T_base) / 1e5
    print('Sound speed: ', cs)
    vrat = 150. / np.minimum(abs(np.sqrt(v_kms**2 + (mach * cs)**2)), cs) 
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

    fig, (ax_mass, ax_vel, ax_tgrow) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True
    )

    run_paths, saveFile = get_working_dirs()
    run_paths = list(run_paths)
    #run_paths = []
    #run_paths.append('/viper/ptmp/ferhi/StratDisk/noturb/r100')

    for j, run in enumerate(run_paths):

        print(f"\n--- {run} ---")

        try:
            sim = StratifiedBox(os.path.join(run, 'strat.in'), dir=run)

            code_time = sim.code_time_cgs
            code_length = sim.code_length_cgs

            t, m = mass_evolution(run)
            _, v = vel_evolution(run)

            if 'noturb' in run:
                Lambda_units = 1
            else:
                Lambda_units = 0.1
            #Lambda_units = float(sim.reader.get('cooling', 'lambda_units_cgs'))

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
        t_myr, v_kms, tgrow_infall, tgrow_vel, tgrow_turb, tcc = \
            compute_tgrow(sim, t, v, m, code_time, code_length, Lambda_units)

        mass_linear = 10**m
        dmdt = np.gradient(mass_linear, t_myr)

        window = int(10)  # choose something reasonable

        kernel = np.ones(window) / window
        dmdt_smooth = np.convolve(dmdt, kernel, mode='same')
        print('This is m / mdot vel infall: ', mass_linear / dmdt_smooth * u.Myr.to('s') * sim.g / 1e5)

        # =========================
        # --- PLOTS ---
        # =========================

        ax_mass.plot(t_myr, m, label=label, alpha=0.8)

        line, = ax_vel.plot(t_myr, v_kms, label=label, alpha=0.8)

        ax_vel.plot(t_myr, tgrow_infall * sim.g/ 1e5,
                    linestyle='-.', color=line.get_color(), alpha=0.8)
        ax_vel.plot(t_myr, mass_linear / dmdt_smooth * u.Myr.to('s') * sim.g / 1e5,
                    linestyle='--', color=line.get_color(), alpha=0.8)
        
        ax_vel.axhline(tgrow_turb * sim.g / 1e5,
                    linestyle='-.', color=line.get_color(), alpha=0.8)

        ax_tgrow.plot(t_myr, mass_linear / dmdt_smooth / (tgrow_infall / u.Myr.to('s')),
                      color=line.get_color(), alpha=0.8, label=label)

        #ax_tgrow.axhline(
        #    tgrow_turb / u.Myr.to('s'),
        #    linestyle='--',
        #    color=line.get_color(),
        #    alpha=0.6,
        #    label=r'$t_{\mathrm{grow}}$' if j == 0 else None
        #)

    # =========================
    # --- FORMAT ---
    # =========================

    print("This is g tgrow: ", tgrow_infall * sim.g / 1e5)
    ax_mass.set_ylabel(r'$\log(m/m_0)$')
    ax_mass.set_ylim(bottom=-2)
    ax_mass.legend()

    ax_vel.set_ylabel('Velocity (km/s)')
    ax_vel.set_ylim(-100, 100)
    ax_vel.legend()

    ax_tgrow.set_xlabel('Time (Myr)')
    ax_tgrow.set_ylabel(r'$m / \dot{m} / tgrow$')
   # ax_tgrow.set_ylim(0, 100)

    plt.tight_layout()

    save_path = f"/u/ferhi/Figures/{saveFile}_cleaned.png"
    print(f"Saved to: {save_path}")
    plt.savefig(save_path)