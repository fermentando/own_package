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


def get_v_mass_at_tff(r0, h, vturb, tcool=None):
    """
    Optimized version - only get final values without interpolation
    """
    if tcool is None: 
        cloud = TurbFallingCloud(h, r0, vturb=vturb)
    else: 
        cloud = TurbFallingCloud(h, r0, vturb=vturb, tcool=tcool)
    
    # FAST: Just get the solution without dense_output if you only need final values
    tcc = 0.9 * 10 * r0 / vturb
    sol = cloud.integrate(tcc, stop_mode='time', dense_output=False)
    
    # Get final values directly from the solution arrays
    z_final = sol.y[0, -1]
    v_final = -sol.y[1, -1]
    m_final = sol.y[2, -1]
    m_initial = sol.y[2, 0]
    print(tcc)
    
    return m_final/m_initial, v_final/1000


def get_v_mass_sim(rundir):
    sim = StratifiedBox(os.path.join(rundir, 'strat.in'), dir=rundir)
    times, norm_mass, cgout, wgout, total = mass_evolution(rundir, gout=True)
    tvs, vel = vel_evolution(rundir)
    mask = ~np.isnan(norm_mass)
    timeseries = times/sim.t_eddy - 6
    idx_0 = np.argmin(np.abs(timeseries))
    norm_mass = norm_mass- norm_mass[idx_0]

    timeseries = timeseries[mask] * sim.t_eddy * sim.code_times_cgs / u.Myr.to('s')
    norm_mass = 10**norm_mass[mask]
    vel = vel[mask]

    #Identify tff
    tcc = 0.9 * 10 * sim.r_cloud_inserted /15 
    idx = np.argmin(np.abs(timeseries - tcc))
    print(timeseries[idx])

    return norm_mass[idx], vel[idx]

def get_tgrow_over_tcc_at_vmax(r0,h,vturb, tcool):
        
    # Run model
    cloud = TurbFallingCloud(h,r0, vturb=vturb, tcool=tcool)
    

    sol = cloud.integrate(0.01, stop_mode='height')

    if len(sol.t_events[0]) > 0:
        t_end = sol.t_events[0][0]
    else:
        t_end = sol.t[-1]

    # Convert to a continuous solution
    t_range = np.linspace(0, t_end, 1000)
    solution = sol.sol(t_range)


    # compute L
    z = solution[0]
    v = -solution[1]
    m = solution[2]

    idx_vmax = np.argmax(v)
    
    
    tgrow_vmax = cloud.vTgrow(z[idx_vmax], v[idx_vmax], m[idx_vmax]) * 1e2 / cloud.profile_g(z[idx_vmax])
    tcc_vmax = np.sqrt(cloud.profile_T(z[idx_vmax])/1e4)*r0 * u.pc.to('m')/v[idx_vmax]

    return tgrow_vmax/tcc_vmax, v[idx_vmax]


if __name__ == "__main__":

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
    simm = []
    simv = []
    comm = []
    comv = []


    radii = [float(r[1:]) for r in r_dirs]


    digits = []
    rclvals = []
    fclst = []
    vmlst = []
    for tdir in t1e_paths:
        sim = StratifiedBox(os.path.join(tdir,'strat.in'), tdir)
        digit = np.log10(sim.compute_restart_cooling_time())

        
        digits.append(digit)
        sm, sv = get_v_mass_sim(tdir)
        simm.append(sm)
        simv.append(sv)
        rclvals.append(sim.r_cloud_inserted)

    digit_min = min(digits)
    digit_max = max(digits)


    logspace_r = np.logspace(min(np.log10(radii) - 0.5), max(np.log10(radii) + 0.1), 30)
    logspace_cooling = np.logspace(digit_min, digit_max, 30)


    combs = list(itertools.product(logspace_r,logspace_cooling))


    # Computations
    for rcl, tcool in tqdm(combs):
        nm, nv = get_v_mass_at_tff(rcl, 3, vturb = 15, tcool=tcool)
        comm.append(np.log10(nm))
        comv.append(nv)
        tt, vm = get_tgrow_over_tcc_at_vmax(rcl,3, vturb=10, tcool=tcool)
        fclst.append(tt)
        vmlst.append(vm)



    import cmasher as cmr
    from matplotlib.colors import LogNorm
    vmin = 1
    vmax = 1e3
    cmap = cmr.ember_r
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # normalization
    mrat = np.reshape(pow(10, np.array(comm)), (len(logspace_r), len(logspace_cooling))).copy().T
    z_m = mrat
    x_m = np.log10(logspace_r/0.7)
    y_m = np.log10(logspace_cooling) + 6
    extent = [x_m.min(), x_m.max(), y_m.min(), y_m.max()]

    plt.style.use('custom_plot')
    plt.figure(figsize=(8,6))

    # imshow with LogNorm
    cnt = plt.imshow(z_m, origin='lower', extent=extent, aspect='auto',
                    cmap=cmap, norm=norm, interpolation='bicubic')

    plt.colorbar(cnt, label=r'$v_{\rm max}$')

    # scatter points
    vals = np.clip(np.array(simm), vmin, vmax)  # clip to the same range
    plt.scatter(np.log10(np.array(rclvals)/0.7), np.array(digits) + 6,
                c=vals, cmap=cmap, norm=norm,
                edgecolors='white', marker='o', s=100) 
    plt.xlabel(r"$\log_{10}(r_{\rm cl}/r_{\rm infall})$")
    plt.ylabel(r"$\log_{10}(z/H_{\rm eff})$")
    plt.grid(visible=False)
    print('\n Figure saved to /u/ferhi/own_package/plots/mass_check_ember.png')
    plt.savefig('/u/ferhi/own_package/plots/mass_check_ember.png')
    plt.clf()

    vmin = 0
    vmax = 20
    cmap = cmr.wildfire
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # normalization
    mrat = np.reshape(comv, (len(logspace_r), len(logspace_cooling))).copy().T
    z_v = mrat
    x_v = np.log10(logspace_r/0.7)
    y_v = np.log10(logspace_cooling) + 6
    extent = [x_v.min(), x_v.max(), y_v.min(), y_v.max()]

    plt.style.use('custom_plot')
    plt.figure(figsize=(8,6))

    # imshow with LogNorm
    cnt = plt.imshow(z_v, origin='lower', extent=extent, aspect='auto',
                    cmap=cmap, norm=norm, interpolation='bicubic')

    plt.colorbar(cnt, label=r'$v \ (km/s)$')

    # scatter points
    vals = np.clip(simv, vmin, vmax)  # clip to the same range
    plt.scatter(np.log10(np.array(rclvals)/0.7), np.array(digits) + 6,
                c=vals, cmap=cmap, norm=norm,
                edgecolors='white', marker='o', s=100) 
    plt.axvline(0, linestyle=':', color='white')
    trat_arr = np.reshape(fclst,(len(logspace_r),len(logspace_cooling))).copy().T
    plt.contour(np.log10(logspace_r/0.7),np.log10(logspace_cooling) + 6,trat_arr,levels=[4],colors='white',linestyles='dashed')
    plt.plot(np.log10(logspace_r/0.7), np.log10(logspace_r/0.7) +5, linestyle= "dashed")

    plt.xlabel(r"$\log_{10}(r_{\rm cl}/r_{\rm infall})$")
    plt.ylabel(r"$\log_{10}(z/H_{\rm eff})$")
    plt.grid(visible=False)


    plt.ylim(bottom=0, top=5)
    print('Figure saved to /u/ferhi/own_package/plots/velocity_check_wildfire.png')
    plt.savefig('/u/ferhi/own_package/plots/velocity_check_wildfire.png')