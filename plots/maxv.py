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
    # 2. Make a list of the radius values (floats) from r* directory names
    # ------------------------------------------------------------
    simv = []
    comv = []
    comv_nodrag = []


    radii = [float(r[1:]) for r in r_dirs]


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

    print(len(rclouds), len(simv))
    

    indices = [i for i, v in enumerate(digits) if -3 < v < -2]
    # After filtering by cooling time indices
    simv = np.array(simv)[indices]
    rclouds = rclouds[indices]
    cooling_times = cooling_times[indices]
    times = np.array(times)[indices]

    # Sort by radius for interpolation
    sort_idx_original = np.argsort(rclouds)
    rclouds_sorted = rclouds[sort_idx_original]
    times_sorted = times[sort_idx_original]

    # ---------------------
    # CREATE EXTENDED RADIUS ARRAY
    # ---------------------
    # Create a denser radius array
    n_points = 10  # Adjust as needed
    r_extended = np.logspace(
        np.log10(rclouds.min()), 
        np.log10(rclouds.max()), 
        n_points
    )

    # Interpolate times based on the radius-time relationship
    time_interpolator = interp1d(
        rclouds_sorted, 
        times_sorted, 
        kind='linear',  # or 'cubic' for smoother interpolation
        fill_value='extrapolate'
    )
    times_extended = time_interpolator(r_extended)

    # ---------------------
    # COMPUTE VELOCITIES FOR EXTENDED ARRAYS
    # ---------------------
    # Choose a fixed cooling time (e.g., log10(tcool) = -2.5)
    target_log_tcool = -2.5

    comv_extended = []
    comv_nodrag_extended = []

    print(f"Computing velocities for {n_points} radii...")
    for rcl, time in tqdm(zip(r_extended, times_extended)):
        nm_drag, nv_drag = get_tgrow_over_tcc_at_vmax(
            rcl, 3, vturb=15, time=time, tcool=10**target_log_tcool
        )
        nm_nodrag, nv_nodrag = get_tgrow_over_tcc_at_vmax(
            rcl, 3, vturb=15, time=time, tcool=10**target_log_tcool, drag_bool=False
        )
        comv_extended.append(nv_drag)
        comv_nodrag_extended.append(nv_nodrag)

    comv_extended = np.array(comv_extended)
    comv_nodrag_extended = np.array(comv_nodrag_extended)

    # ---------------------
    # PLOTTING
    # ---------------------
    # Original scattered simulation data
    plt.errorbar(
        rclouds/0.7, simv,
        yerr=5,
        fmt='o',
        color='tab:blue',
        ecolor='gray',
        elinewidth=1.2,
        capsize=3,
        markersize=7,
        markeredgecolor='black',
        label='Simulation data'
    )

    # Extended turbulence line
    plt.plot(
        r_extended/0.7,
        comv_extended,
        color='black',
        linestyle='--',
        linewidth=2,
        label=r"turbulence"
    )

    # Extended no turbulence line
    plt.plot(
        r_extended/0.7,
        comv_nodrag_extended,
        color='black',
        linestyle=':',
        linewidth=2,
        label=r"no turbulence"
    )

    plt.xlabel(r'$\log_{10}(r_{\rm cl} / r_{\rm infall})$')
    plt.ylabel(r'$v$ (km s$^{-1}$)')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('v_r_plot.png')


