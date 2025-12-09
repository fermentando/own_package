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



def get_tgrow_over_tcc_at_vmax(r0,h,vturb, drag_bool = True):
        
    # Run model
    cloud = TurbFallingCloud(h,r0, vturb=vturb)

    def in_safe_zone(h):
        pressure =  cloud.profile_n(h) * cloud.profile_T(h)
        safe_pressure = 3000*(cloud.profile_g(h)/(1e-8))**0.8 * ((cloud.profile_T(h)/1e6))**(12/5)
        return (pressure > safe_pressure)
    

    sol = cloud.integrate(0.01, stop_mode='height', drag_bool= drag_bool)

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

    idx_cut = np.argmin(np.abs(m/m[0] - 1000.))

    z = z[:idx_cut]
    v = v[:idx_cut]
    m = m[:idx_cut]

    idx_vmax = np.argmax(v)

    if in_safe_zone(h):
        return 0, v[idx_vmax]
    
    
    tgrow_vmax = cloud.vTgrow(z[idx_vmax], v[idx_vmax], m[idx_vmax]) * 1e2 / cloud.profile_g(z[idx_vmax])
    tcc_vmax = np.sqrt(cloud.profile_T(z[idx_vmax])/1e4)*r0 * u.pc.to('m')/v[idx_vmax]

    return tgrow_vmax/tcc_vmax, v[idx_vmax]/1000


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


    comv_extended = []
    comv_nodrag_extended = []

    radius = np.logspace(-1, 2, 30)

    for rcl in tqdm(radius):
        nm_drag, nv_drag = get_tgrow_over_tcc_at_vmax(
            rcl, 60, vturb=15
        )
        nm_nodrag, nv_nodrag = get_tgrow_over_tcc_at_vmax(
            rcl, 60, vturb=15, drag_bool=False
        )
        comv_extended.append(nv_drag)
        comv_nodrag_extended.append(nv_nodrag)

    comv_extended = np.array(comv_extended)
    comv_nodrag_extended = np.array(comv_nodrag_extended)



    # Extended turbulence line
    plt.plot(
        radius/0.7,
        comv_extended,
        color='black',
        linestyle='--',
        linewidth=2,
        label=r"turbulence"
    )

    # Extended no turbulence line
    plt.plot(
        radius/0.7,
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
    plt.savefig('vdiff.png')


