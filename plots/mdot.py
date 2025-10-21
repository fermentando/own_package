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

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{cancel}"
})
# Set up a colormap using seaborn
cmap = sns.color_palette("rocket", as_cmap=True)  # or "magma", "plasma", etc.
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
        correction = 0
        for i in range(1, len(vboost)):
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
        timeseries = data[:, 0]
        
        return timeseries, delta_v
    
        
    
if __name__ == "__main__":
    
    
    plot_yt = False
    plot_hst = True

    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(1,2, width_ratios=[1, 0.03], wspace=0.1, hspace=0.1, figure=fig)
    ax = np.empty((2, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])


    
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    run_paths = ['/viper/ptmp/ferhi/LEGACY/fvLism/02kc','/viper/ptmp/ferhi/LEGACY/fvLism/01kc']



    for j, run in enumerate(run_paths):
        all_runs = glob.glob(os.path.join(run, 'fv*'))
        if 'scaleless' in run:
            all_runs = [run]
        #if j == 1:
            #other_dirs = glob.glob('/viper/ptmp/ferhi/d40rcl/01kc/fv*')
            #all_runs.extend(other_dirs)
            #more_dirs = glob.glob('/viper/ptmp/ferhi/d80rcl/01ekc/fv*')
            #all_runs.extend(more_dirs)
        #if j ==0: 
            #other_dirs = glob.glob('/viper/ptmp/ferhi/d20rcl/02ekc/fv*')
            #all_runs.extend(other_dirs)
        for run_name in all_runs:
            if "/viper/ptmp/ferhi/LEGACY/d20rcl/02ekc/fv03_lowres_raven" in run_name: continue
            if "fv01_longer" in run_name: continue
            if run_name.split('/')[-1] == 'fv01_scaleless': continue
                
            sim = SingleCloudCC(os.path.join(run_name, 'ism.in'), dir=run_name)
            code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
            code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
            files = np.sort(glob.glob(os.path.join(run_name, 'out/parthenon.prim.*.phdf')))
            depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
            fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
            base_fv = int(-np.log10(fv))
            dt = float(sim.reader.get('parthenon/output1', 'dt'))
            dt2 = float(sim.reader.get('parthenon/output0', 'dt'))
            rho_wind = float(sim.reader.get('problem/wtopenrun', 'rho_wind_cgs'))
            rho_cloud = float(sim.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
            chi = rho_cloud / rho_wind
            kin = sim.reader.get('problem/wtopenrun', 'kmin')
            sigma = tuple(float(kin) for i in range(2))  if ',' not in kin else tuple(float(k) for k in kin.split(','))
            t_cool_min = get_t_cool_min(sim.rho_cloud, sim.T_cloud, sim.mbar)
            print("This is ratio of tcools: ",t_cool_min/sim.tcoolmix)
            
            tsh =  depth * sim.R_cloud / sim.v_wind

            t1 = 0.1 * chi**0.5 * sim.R_cloud / sim.v_wind
            t2 =  chi**0.5 * fv * depth *  sim.R_cloud / sim.v_wind

            # Linear sum
            t_linear = t1 + t2
            tcool = get_t_cool_cgs(sim.rho_cloud, sim.T_cloud, sim.mbar)
            tcoolmin = get_t_cool_min(sim.rho_cloud, sim.T_cloud, sim.mbar)
            cs = get_c_s(sim.T_wind)
            tsc =  (0.1 * sigma[-1] )* chi**0.5 * sim.R_cloud / sim.v_wind
            tcooleff = (tsc * tcool) ** 0.5
            tgrow =  100*np.sqrt( tsc * sim.tcoolmix)
            #print(f"t_linear: {t_linear:.3e}, tcc: {0.1*sim.tcc:.3e}, tcoolmix: {sim.tcoolmix:.3e}")
            
            
            timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run_name, gout)

            #mask = ~np.isnan(norm_mass)
            #norm_mass = norm_mass[mask]
            #timeseries = timeseries[mask]
            color = sm.to_rgba(10*depth)

            label = run.split('/')[-1]
            plt.style.use('custom_plot') 

            time_axis = timeseries * code_time_cgs 
            idx = (np.abs(time_axis - tsh)).argmin()


            t_shift = 0.65 * tsh if j ==1 else 0
            correction_index_mass = (np.abs(time_axis - t_shift)).argmin()

            x_mass = (timeseries * code_time_cgs - t_shift) 
            y_mass = norm_mass - norm_mass[correction_index_mass]
            #y_mass *= y_mass[0]
            #y_mass = np.convolve(y_mass, np.ones(30)/30, mode='same')

            # Compute numerical derivative
            dy_dx_mass = np.gradient(y_mass, x_mass)
            if np.any(dy_dx_mass > 0.3):
                print(f"Warning: Positive derivative detected in {run_name} at fv = {fv}")
            #ax[0,0].plot(x_mass, dy_dx_mass * x_mass, 
            #    color=color, linestyle=linestyles[base_fv], alpha=0.8)
            if 'scaleless' in run or 'scaleless' in run_name:
                ax[0,0].plot(x_mass/t_linear, dy_dx_mass  * tgrow,
                         color="black", linestyle='-', alpha=0.8)
            else:
                ax[0,0].plot(x_mass/t_linear, dy_dx_mass * tgrow,
                         color=color, linestyle=linestyles[base_fv], alpha=0.8)

            #ax[0, j].plot(time_axis[idx]/t_linear, norm_mass[idx], marker='o', color='black', zorder=10, alpha = 0.8)  
            #ax[1, j].plot(time_axis2[idx2]/t_linear, v_normalised[idx2], marker='o', color='black', zorder=10, alpha = 0.8)
            

 


fv_legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label=r'$f_v = 10^{\mathrm{-1}}$'),
    Line2D([0], [0], color='black', linestyle=':', label=r'$f_v = 10^{\mathrm{-2}}$'),
    Line2D([0], [0], color='black', linestyle='--', label=r'$f_v = 10^{\mathrm{-3}}$'),
]
fig.subplots_adjust(top=0.88)  
fig.legend(
    handles=fv_legend_elements,
    loc='upper center',
    ncol=3,
    bbox_to_anchor=(0.5, 1.005),  # Slightly above the plot
    frameon=True,
)

cl_legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label=r'$\cancel{r_\mathrm{cl}} $')
]
ax[0,0].legend(
    handles=cl_legend_elements,
    loc='upper right',  # Change as needed
    frameon=True,
)

cax = fig.add_subplot(gs[:, 1])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label(r'$L_{\mathrm{ISM}}$ [$r_{\mathrm{cl}}$]')
cax.tick_params(axis='y', which='both', color='white', labelcolor='black', direction='in')

# Axis labels
#ax[0, 0].set_ylabel(r'$\log\left(m(T < 2T_\mathrm{cl}) / m_0\right)$', labelpad=8)
ax[0,0].set_ylim(bottom=1e-2)
ax[0,0].set_xlim(left=0, right = 40)
ax[0,0].set_yscale('log')
ax[0,0].set_xlabel(r'$\tau$')
ax[0,0].set_ylabel(r'$\dot{m} \,  t_\mathrm{grow} / m_\mathrm{0}$', labelpad=5)

    


print('Saved to: ', '/u/ferhi/Figures/tgrow_mdot.pdf')
plt.savefig('/u/ferhi/Figures/tgrow_mdot.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()
