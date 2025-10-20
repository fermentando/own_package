import os
import yt 
import glob
import sys
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import argparse
from read_hdf5 import read_hdf5
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


# Set up a colormap using seaborn
cmap = sns.color_palette("mako", as_cmap=True)  # or "magma", "plasma", etc.
norm = mcolors.LogNorm(vmin=10, vmax=1e4)  # Log scale if range is wide
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.style.use('custom_plot')
linestyles = {1:'-', 3:'--', 0:'-.', 2:':'}

def hst_evolution(run, gout=False):
        
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        data = np.where(data==0, 1e-22, data)
        if np.shape(data)[1] >= 17: mass_ind = 10
        else: mass_ind = 10
        norm_mass = np.log10(data[:, mass_ind]/data[0, mass_ind])
        timeseries = data[:, 0]
        
        wgout = np.zeros_like(timeseries); cgout = wgout
        sum = norm_mass
        
        if gout: 
            wgout = np.log10(data[:, -2]/data[0, mass_ind]) 
            cgout = np.log10(data[:, -3]/data[0, mass_ind])
            sum = np.log10((data[:, mass_ind]+data[:, -2]+data[:, -3])/data[0, mass_ind])
        return timeseries, norm_mass, cgout, wgout, sum

def hst_entrainment(run, vwind):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        vboost = data[:, -1]
        if np.shape(data)[1] >= 17:
            mass = data[:,11]
            vx2 = abs(data[:,13])/(mass)
        else: 
            mass = data[:,10]
            vx2 = abs(data[:,12])/(mass)
        delta_v = (vwind - (vx2 + vboost))/vwind
        timeseries = data[:, 0]
        
        return timeseries, delta_v
    
        
    
if __name__ == "__main__":
    
    
    plot_yt = False
    plot_hst = True

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 0.08, 1], wspace=0.4, figure=fig)
    ax = np.array([fig.add_subplot(gs[0]), fig.add_subplot(gs[2])])

    
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    run_paths = ['/viper/ptmp/ferhi/fvLism/01kc/fv01_scaleless/','/viper/ptmp/ferhi/fvLism/AGN']



    for j, run_name in enumerate(run_paths):

            
        sim = SingleCloudCC(os.path.join(run_name, 'ism.in'), dir=run_name)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        files = np.sort(glob.glob(os.path.join(run_name, 'out/parthenon.prim.*.phdf')))
        depth = 10 * float(sim.reader.get('problem/wtopenrun', 'depth'))
        rhow = float(sim.reader.get('problem/wtopenrun', 'rho_wind_cgs'))
        rhoc = float(sim.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
        chi = rhoc/rhow
        fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
        base_fv = int(-np.log10(fv))
        
        
        tccfact =  depth #if sim.tcoolmix/sim.tcc >= 0.1 else 0.1
        #tsh =  fv * depth * sim.R_cloud / sim.v_wind #np.sqrt( fv**-0.6 + 100)
        tsh =  (fv**-(1/3) + chi) * 0.1 * sim.R_cloud / sim.v_wind if sim.tcoolmix/sim.tcc >= 0.1 else 0.1 * sim.tcc

        print(run_name)
        try:
            timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run_name, gout)
            v_wind = sim.v_wind / code_length_cgs * code_time_cgs
            times, v_normalised = hst_entrainment(run_name, vwind=v_wind)
        except Exception as e:
            continue
        mask = ~np.isnan(norm_mass)
        norm_mass = norm_mass[mask]
        timeseries = timeseries[mask]
        color = sm.to_rgba(tccfact)


        plt.style.use('custom_plot')
        ax[0].plot(timeseries * code_time_cgs / tsh, norm_mass, color=f'C{j}', linestyle=linestyles[base_fv], alpha=0.8)
        ax[1].plot(times * code_time_cgs / tsh, v_normalised, color=f'C{j}', linestyle=linestyles[base_fv], alpha=0.8)

 



# Axis labels
ax[0].set_ylabel(r'$\log\left(m(T < 2T_\mathrm{cl}) / m_0\right)$', labelpad=8)
ax[1].set_ylabel(r'$\Delta v_\mathrm{shear} / v_w$', labelpad = 8)
ax[0].set_ylim(bottom=-3, top=0)
ax[0].set_ylim(bottom=0, top=1)
for axs in [ax[0], ax[1]]:
    axs.set_xlabel(r'$t  [\tilde{t}_{cc}]$')


plt.savefig('/u/ferhi/Figures/scaleless_chi.png', dpi = 300, bbox_inches = 'tight')
plt.show()
