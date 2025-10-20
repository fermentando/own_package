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
import h5py


# Set up a colormap using seaborn
cmap = sns.color_palette("seismic", as_cmap=True)  # or "magma", "plasma", etc.
norm = mcolors.Normalize(vmin=0, vmax=3)  # Log scale if range is wide
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.style.use('custom_plot')
linestyles = {1:'-', 3:'--', 0:'-.', 2:':'}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{cancel}"
})

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

def hst_entrainment(run, vwind, threshold=0.2):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        vboost = data[:, -1]
        correction = 0

        for i in range(10, len(vboost)):
            if vboost[i] < 0.5 and vboost[i-1] > 1.0:  # heuristically detect a reset
                correction = vboost[i-1]
            vboost[i] += correction
        if np.shape(data)[1] >= 17:
            mass = data[:,11]
            vx2 = abs(data[:,13])/(mass)
        else: 
            mass = data[:,10]
            vx2 = abs(data[:,12])/(mass)
        delta_v = (vwind - (vx2 + vboost))/vwind

        restart_fix = 0
        diffs = np.diff(delta_v)
        jumps = np.where(np.abs(diffs) > threshold)[0]
        for j in jumps:
            restart_fix += delta_v[j] - delta_v[j+1]
            delta_v[j+1:] += restart_fix
        timeseries = data[:, 0]
        
        return timeseries, delta_v
    
        
    
if __name__ == "__main__":
    
    
    plot_yt = False
    plot_hst = True

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.14, 1], wspace=0.1, hspace=0.14, figure=fig)
    ax = np.empty((2, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 2])



    
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    run_paths = ['/viper/ptmp/ferhi/fvLism/01kc/fv01_scaleless', '/viper/ptmp/ferhi/fvLism/01kc/fv01_scaleless_mach2', '/viper/ptmp/ferhi/fvLism/correct_M_overdense','/viper/ptmp/ferhi/fvLism/overdense/fv01_longer', '/viper/ptmp/ferhi/fvLism/01kc/scaleless_destruction']
    legends = [
         #r'$\cancel{r_\mathrm{cl}}$',
         r'$v_w=440 \, $kms$^{-1}$',
         r'$\chi=10^3$',
         r'$\chi=10^3, v_w=725 \, $kms$^{-1}$'
    ]


    for j, run_name in enumerate(run_paths):

            
        sim = SingleCloudCC(os.path.join(run_name, 'ism.in'), dir=run_name)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        files = np.sort(glob.glob(os.path.join(run_name, 'out/parthenon.prim.*.phdf')))
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
        base_fv = int(-np.log10(fv))
        dt = float(sim.reader.get('parthenon/output1', 'dt'))
        dt2 = float(sim.reader.get('parthenon/output0', 'dt'))

        
        
        tsh =  depth * sim.R_cloud / sim.v_wind

        t1 = sim.R_cloud / sim.v_wind
        t2 = 10 * fv * depth *  sim.R_cloud / sim.v_wind

        # Linear sum
        t_linear = t1 + t2

        try:
            timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run_name, gout)
            v_wind = sim.v_wind / code_length_cgs * code_time_cgs
            times, v_normalised = hst_entrainment(run_name, vwind=v_wind)
        except Exception as e:
            continue
        #mask = ~np.isnan(norm_mass)
        #norm_mass = norm_mass[mask]
        #timeseries = timeseries[mask]
        color = sm.to_rgba(10*depth)
        

       
        plt.style.use('custom_plot') 

        time_axis = timeseries * code_time_cgs 
        time_axis2 = times * code_time_cgs 
        idx = (np.abs(time_axis - tsh)).argmin()
        idx2 = (np.abs(time_axis2 - tsh)).argmin()

        t_shift = 0.65 * tsh if j ==1 else 0
        correction_index_mass = (np.abs(time_axis - t_shift)).argmin()
        correction_index_v = (np.abs(time_axis2 - t_shift)).argmin()

        if j == 0:
            ax[0,0].plot((timeseries * code_time_cgs - t_shift) / t_linear, norm_mass-norm_mass[correction_index_mass], color = cmap(norm(j)), alpha=0.8)
            ax[0,1].plot((times * code_time_cgs - t_shift)/ t_linear, v_normalised, color = cmap(norm(j)), alpha=0.8)
        elif j == len(run_paths)-1:
            ax[0,0].plot((timeseries * code_time_cgs - t_shift) / t_linear, norm_mass-norm_mass[correction_index_mass], color = 'black', alpha=0.8)
            ax[0,1].plot((times * code_time_cgs - t_shift)/ t_linear, v_normalised, color = 'black', alpha=0.8)
        else:
            label = legends[j-1]
            print(j)
            ax[0,0].plot((timeseries * code_time_cgs - t_shift) / t_linear, norm_mass-norm_mass[correction_index_mass], label = label, color = cmap(norm(j)), alpha=0.8)
            ax[0,1].plot((times * code_time_cgs - t_shift)/ t_linear, v_normalised, label = label, color = cmap(norm(j)), alpha=0.8)
        #ax[0, j].plot(time_axis[idx]/t_linear, norm_mass[idx], marker='o', color='black', zorder=10, alpha = 0.8)  
        #ax[1, j].plot(time_axis2[idx2]/t_linear, v_normalised[idx2], marker='o', color='black', zorder=10, alpha = 0.8)
        
        ax[0,1].set_xlim(left = 0, right = 30)




# Axis labels
ax[0, 0].legend(frameon=True, fontsize=10, loc='lower right', prop={'size': 9.5})
ax[0, 0].set_ylabel(r'$\log\left(m(T < 2T_\mathrm{cl}) / m_0\right)$', labelpad=8, fontsize=16)
ax[0, 1].set_ylabel(r'$\Delta v_\mathrm{shear} / v_w$', labelpad = 8, fontsize=16)
for axs in [ax[0, 0], ax[0, 1]]:
    axs.set_xlabel(r'$\tau$', fontsize=16)
    axs.set_xlim(left = 0, right = 40)
    


ax[0,0].set_ylim(-3, 1)
print("Saving figure to /u/ferhi/Figures/other_cases_evolution.png")
plt.savefig('/u/ferhi/Figures/other_cases_evolution.png', dpi = 300, bbox_inches = 'tight')
plt.show()
