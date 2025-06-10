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

    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.08], wspace=0.1, hspace=0.1, figure=fig)
    ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]).reshape(2, 2)

    
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    run_paths = ['/viper/ptmp/ferhi/fvLism/02kc','/viper/ptmp/ferhi/fvLism/01kc']



    for j, run in enumerate(run_paths):
        all_runs = glob.glob(os.path.join(run, 'fv*'))
        if j == 1:
            other_dirs = glob.glob('/viper/ptmp/ferhi/d40rcl/01kc/fv*')
            all_runs.extend(other_dirs)
            more_dirs = glob.glob('/viper/ptmp/ferhi/d80rcl/01ekc/fv*')
            all_runs.extend(more_dirs)
        if j ==0: 
            other_dirs = glob.glob('/viper/ptmp/ferhi/d20rcl/02ekc/fv*')
            all_runs.extend(other_dirs)
        for run_name in all_runs:
            if "/viper/ptmp/ferhi/d20rcl/02ekc/fv03_lowres_raven" in run_name: continue
            if "scaleless" in run_name: continue
                
            sim = SingleCloudCC(os.path.join(run_name, 'ism.in'), dir=run_name)
            code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
            code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
            files = np.sort(glob.glob(os.path.join(run_name, 'out/parthenon.prim.*.phdf')))
            depth = 10 * float(sim.reader.get('problem/wtopenrun', 'depth'))
            base_fv = int(run_name.split('fv')[-1][:2])
            fv = 10 ** (-base_fv)
            
            
            tccfact =  depth #if sim.tcoolmix/sim.tcc >= 0.1 else 0.1
            #tsh =  fv * depth * sim.R_cloud / sim.v_wind #np.sqrt( fv**-0.6 + 100)
            tsh =  (fv**-(1/3) + 10) * 0.1 * sim.R_cloud / sim.v_wind if sim.tcoolmix/sim.tcc >= 0.1 else 0.1 * sim.tcc
            tsh = depth * fv * sim.R_cloud / sim.v_wind 
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


            label = run.split('/')[-1]
            plt.style.use('custom_plot')
            ax[0,j].plot(timeseries * code_time_cgs / tsh, norm_mass, color=color, linestyle=linestyles[base_fv], alpha=0.8)
            ax[1,j].plot(times * code_time_cgs / tsh, v_normalised, color=color, linestyle=linestyles[base_fv], alpha=0.8)

 


fv_legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label=r'$f_v = 10^{\mathrm{-1}}$'),
    Line2D([0], [0], color='black', linestyle=':', label=r'$f_v = 10^{\mathrm{-2}}$'),
    Line2D([0], [0], color='black', linestyle='--', label=r'$f_v = 10^{\mathrm{-3}}$'),
]
fig.subplots_adjust(top=0.93)  
fig.legend(
    handles=fv_legend_elements,
    loc='upper center',
    ncol=3,
    bbox_to_anchor=(0.5, 1.01),  # Slightly above the plot
    frameon=True,
)

cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label(r'$L_{\mathrm{ISM}}$ [$r_{\mathrm{cl}}$]')
cax.tick_params(axis='y', which='both', color='white', labelcolor='black', direction='in')

# Axis labels
ax[0, 0].set_ylabel(r'$\log\left(m(T < 2T_\mathrm{cl}) / m_0\right)$', labelpad=8)
ax[1, 0].set_ylabel(r'$\Delta v_\mathrm{shear} / v_w$', labelpad = 8)
for axs in [ax[1, 0], ax[1, 1]]:
    axs.set_xlabel(r'$t  [\tilde{t}_{cc}]$')

for axs in [ax[0, 0], ax[0, 1]]:
    axs.set_ylim(-3, 1)

plt.setp(ax[0,1].get_yticklabels(), visible=False)
plt.setp(ax[1,1].get_yticklabels(), visible=False)
plt.setp(ax[0,0].get_xticklabels(), visible=False)
plt.setp(ax[0,1].get_xticklabels(), visible=False)

plt.savefig('/u/ferhi/Figures/comparative_mass_evolution.png', dpi = 300, bbox_inches = 'tight')
plt.show()
