from infalling_clouds import *
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as C
import latexify
import os
from flow_props import mass_evolution, vel_evolution, StratifiedBox
from tqdm import tqdm
import itertools
from scipy.interpolate import interp1d



def get_tgrow_over_tcc_at_vmax(r0, h, vturb, time, tcool, drag_bool=True):
    

    cloud = TurbFallingCloud(h, r0, vturb=vturb, tcool=tcool)
    sol = cloud.integrate(time, stop_mode='time', drag_bool=drag_bool, dense_output=False)

    z_final = sol.y[0, -1]
    v_final = -sol.y[1, -1]
    m_final = sol.y[2, -1]



    # ---- 5. Compute tgrow and tcc safely ----
    try:
        tgrow_vmax = cloud.vTgrow(z_final, v_final, m_final) * 1e2 / cloud.profile_g(z_final)
        tcc_vmax = np.sqrt(cloud.profile_T(z_final) / 1e4) * r0 * u.pc.to("m") / v_final
    except Exception:
        return np.nan, np.nan

    return tgrow_vmax / tcc_vmax, v_final/1000


def get_v_max_sim(rundir):
    sim = StratifiedBox(os.path.join(rundir, 'strat.in'), dir=rundir)
    times, vel = vel_evolution(rundir)
    mask = ~np.isnan(vel)
    vel = vel[mask]
    timeseries = times/sim.t_eddy - 6
    timeseries = timeseries[mask] * sim.t_eddy * sim.code_times_cgs / u.Myr.to('s')

    return  timeseries[np.argmax(vel)], vel[np.argmax(vel)]


if __name__ == "__main__":

    
    plt.style.use('custom_plot')
    plt.figure(figsize=(8,6))

    base_dir = "/viper/ptmp/ferhi/StratDisk/Rsys/m0.1/"

    # ------------------------------------------------------------
    # 1. List all subdirectories starting with "t1e" inside any r* directory
    #    and also list the r* directories (whose names encode a float after 'r')
    # ------------------------------------------------------------
    r_dirs = []

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("r"):
            try:
                float(entry[1:])
                r_dirs.append(entry)
            except ValueError:
                pass

    t1e_paths = []   # full paths
    for r in r_dirs:
        r_path = os.path.join(base_dir, r)
        for entry in os.listdir(r_path):
            path = os.path.join(r_path, entry)
            if os.path.isdir(path) and entry.startswith("t1e"):
                t1e_paths.append(path)

    # ------------------------------------------------------------
    # 2. Gather data from all simulations
    # ------------------------------------------------------------
    simv = []
    digits = []
    rclvals = []
    times = []

    for tdir in t1e_paths:
        sim = StratifiedBox(os.path.join(tdir,'strat.in'), tdir)
        digit = np.log10(sim.compute_restart_cooling_time())   
        digits.append(digit)
        time, mv = get_v_max_sim(tdir)
        simv.append(mv)
        rclvals.append(sim.r_cloud_inserted)
        times.append(time)

    cooling_times = np.asarray(digits)
    rclouds = np.asarray(rclvals)
    simv = np.array(simv)
    times = np.array(times)

    print(f"Total simulations: {len(rclouds)}")

    # ============================================================
    # Fixed radius, varying cooling time
    # ============================================================
    # Filter by radius
    target_radius = 100
    radius_tolerance = 0.1 * target_radius
    indices = [i for i, r in enumerate(rclouds) 
            if abs(r - target_radius) < radius_tolerance]

    simv_filtered = simv[indices]
    rclouds_filtered = rclouds[indices]
    cooling_times_filtered = cooling_times[indices]
    times_filtered = times[indices]

    # Sort by cooling time
    sort_idx = np.argsort(cooling_times_filtered)
    cooling_times_sorted = cooling_times_filtered[sort_idx]
    times_sorted = times_filtered[sort_idx]
    simv_sorted = simv_filtered[sort_idx]
    rclouds_sorted = rclouds_filtered[sort_idx]

    print(f"Found {len(cooling_times_sorted)} simulations near radius {target_radius:.3f}")

    # Compute predictions for each data point
    comv = []
    comv_nodrag = []

    print(f"Computing velocities for {len(cooling_times_sorted)} points...")
    for log_tcool, time, rcl in tqdm(zip(cooling_times_sorted, times_sorted, rclouds_sorted)):
        nm_drag, nv_drag = get_tgrow_over_tcc_at_vmax(
            rcl, 3, vturb=15, time=time, tcool=10**log_tcool
        )
        nm_nodrag, nv_nodrag = get_tgrow_over_tcc_at_vmax(
            rcl, 3, vturb=15, time=time, tcool=10**log_tcool, drag_bool=False
        )
        comv.append(nv_drag)
        comv_nodrag.append(nv_nodrag)

    comv = np.array(comv)
    comv_nodrag = np.array(comv_nodrag)

    # Plot simulation data
    plt.errorbar(
        cooling_times_sorted, simv_sorted,
        yerr=5, fmt='o', color='tab:blue', ecolor='gray',
        elinewidth=1.2, capsize=3, markersize=7,
        markeredgecolor='black', label='Simulation data'
    )

    # Plot turbulence prediction
    plt.plot(
        cooling_times_sorted, comv,
        color='black', linestyle='--', linewidth=2,
        label=r"turbulence"
    )

    # Plot no turbulence prediction
    plt.plot(
        cooling_times_sorted, comv_nodrag,
        color='black', linestyle=':', linewidth=2,
        label=r"no turbulence"
    )

    plt.xlabel(r'$\log_{10}(t_{\rm cool})$ (log yr)')
    plt.ylabel(r'$v$ (km s$^{-1}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'Fixed radius: $r_{{\\rm cl}} \\approx {target_radius:.1f}$')
    plt.savefig('vmax_tcool.png')


