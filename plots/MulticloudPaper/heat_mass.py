import os
import yt 
import glob
import numpy as np
from utils import *
from adjust_ics import *
from single_cloud import SingleCloudCC
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
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

    fig = plt.figure(figsize=(7,5))


    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    run_paths = ['/viper/ptmp/ferhi/LEGACY/fvLism/01kc/fv01_30r','/viper/ptmp/ferhi/LEGACY/fvLism/01kc/fv01_30r_noTceil', 
                '/viper/ptmp/ferhi/LEGACY/fvLism/02kc/fv01', '/viper/ptmp/ferhi/LEGACY/fvLism/02kc/fv01_noTceil']



    for j, run_name in enumerate(run_paths):
        #if j == 1:
            #other_dirs = glob.glob('/viper/ptmp/ferhi/d40rcl/01kc/fv*')
            #all_runs.extend(other_dirs)
            #more_dirs = glob.glob('/viper/ptmp/ferhi/d80rcl/01ekc/fv*')
            #all_runs.extend(more_dirs)
        #if j ==0: 
            #other_dirs = glob.glob('/viper/ptmp/ferhi/d20rcl/02ekc/fv*')

            sim = SingleCloudCC(os.path.join(run_name, 'ism.in'), dir=run_name)
            code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
            code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
            files = np.sort(glob.glob(os.path.join(run_name, 'out/parthenon.prim.*.phdf')))
            depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
            base_fv = int(run_name.split('fv')[-1][:2])
            fv = 10 ** (-base_fv)
            dt = float(sim.reader.get('parthenon/output1', 'dt'))
            dt2 = float(sim.reader.get('parthenon/output0', 'dt'))
            
            
            tsh =  depth * sim.R_cloud / sim.v_wind

            t1 = sim.R_cloud / sim.v_wind
            t2 = 10 * fv * depth *  sim.R_cloud / sim.v_wind

            # Linear sum
            t_linear = t1

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

            if j==0:
                 label = r'$\Lambda_{\mathrm{ceil}}$'
                 color = "#175ecf"
                 linestyles = '-'
            if j==1: 
                label = r'no $\Lambda_{\mathrm{ceil}}$'
                color = "#175ecf"
                linestyles = '--'

            if j ==2:
                color = "#f7be10"
                label = ''
                linestyles = '-'
            if j == 3:
                color = "#f7be10"
                label = ''
                linestyles = '--'
            plt.style.use('custom_plot') 

            time_axis = timeseries * code_time_cgs 
            time_axis2 = times * code_time_cgs 
            idx = (np.abs(time_axis - tsh)).argmin()
            idx2 = (np.abs(time_axis2 - tsh)).argmin()
            if "fv03" in run_name:
                print(time_axis[idx] / t_linear, idx2)

            t_shift = 0.65 * tsh #if j ==1 else 0
            correction_index_mass = (np.abs(time_axis - t_shift)).argmin()
            correction_index_v = (np.abs(time_axis2 - t_shift)).argmin()


            plt.plot((timeseries * code_time_cgs - t_shift) / t_linear, norm_mass-norm_mass[correction_index_mass], color=color, alpha=0.8, label=label, linestyle=linestyles)
            #ax[0, j].plot(time_axis[idx]/t_linear, norm_mass[idx], marker='o', color='black', zorder=10, alpha = 0.8)  
            #ax[1, j].plot(time_axis2[idx2]/t_linear, v_normalised[idx2], marker='o', color='black', zorder=10, alpha = 0.8)
            
            plt.xlim(left = 0)


# Define line styles
line_styles = ["--", "-"]
chi_labels = [r'no $\Lambda_{\mathrm{ceil}}$', r'$\Lambda_{\mathrm{ceil}}$']

# Create legend handles for line styles
line_handles = [Line2D([0], [0], color='k', linestyle=ls, label=label) for ls, label in zip(line_styles, chi_labels)]

# Combine both handles
all_handles = line_handles

# Axis labels
plt.ylabel(r'$\log\left(m(T < 2T_\mathrm{cl}) / m_0\right)$', labelpad=8)

plt.xlabel(r'$\tau$')
plt.legend(handles = all_handles, loc="lower right")
plt.ylim(-3, 1)



print("Saving figure to /u/ferhi/Figures/heating_convergence.pdf")
plt.savefig('/u/ferhi/Figures/heating_convergence.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()
