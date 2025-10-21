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



original_cmap = plt.cm.BrBG


# Define colors (original)
purple = np.array([106/255, 13/255, 173/255])
light_center = np.array([245/255, 245/255, 245/255])
orange = np.array([255/255, 140/255, 0/255])

N = 256
half = N // 2

# Gradients for inverted colormap: orange -> light center -> purple
left = np.linspace(orange, light_center, half)
right = np.linspace(light_center, purple, half)

colors = np.vstack((left, right))

inverted_cmap = mcolors.LinearSegmentedColormap.from_list("OrangeLightPurple", colors)


# Create a new colormap from modified colors
black_mid_cmap = mcolors.LinearSegmentedColormap.from_list('BrBG_black_mid', colors)
cmap = sns.color_palette("managua", as_cmap=True)  # or "magma", "plasma", etc.
norm = mcolors.Normalize(vmin=10, vmax=80)  # Log scale if range is wide
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

    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.15, 1, 0.06], wspace=0.15,  figure=fig)
    ax = np.empty((1, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 2])


    
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    run_paths = [
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/10chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/20chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/30chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/40chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/50chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/half40chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/half50chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/half60chi',
        '/viper/ptmp/ferhi/LEGACY/fvLism/0.1d_crit/half_single_cloud',
                 
                 ]



    for j, run_name in enumerate(run_paths):

        if "/viper/ptmp/ferhi/d20rcl/02ekc/fv03_lowres_raven" in run_name: continue
        if "scaleless" in run_name: continue
            
        sim = SingleCloudCC(os.path.join(run_name, 'ism.in'), dir=run_name)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        files = np.sort(glob.glob(os.path.join(run_name, 'out/parthenon.prim.*.phdf')))
        try:
            distance = float(run_name.split('/')[-1].split('chi')[0][-2:])
            print("this is separation between cloudlets: ", distance)

        except: 
             distance = 0
        depth = sim.R_cloud * (1 + distance) * 2
        
        
        tsh =  depth * sim.R_cloud / sim.v_wind

        t1 = sim.R_cloud / sim.v_wind
        t2 = 10 * (2*distance)/sim.R_cloud * depth *  sim.R_cloud / sim.v_wind

        # Linear sum
        t_linear = sim.tcc

        try:
            timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run_name, gout)
            v_wind = sim.v_wind / code_length_cgs * code_time_cgs
            times, v_normalised = hst_entrainment(run_name, vwind=v_wind)
        except Exception as e:
            continue

        
        color = sm.to_rgba(distance)

        #label = run_name.split('/')[-1].split('chi')[0][-2:] + r'$ \r_\mathrm{cl}$'
        plt.style.use('custom_plot') 
        if 'single' in run_name: linestyle = linestyles[3]; color = 'grey'
        elif 'half' in run_name: linestyle = linestyles[1]
        else: linestyle = linestyles[3]





        ax[0,0].plot((timeseries * code_time_cgs) / t_linear, norm_mass, color=color, linestyle=linestyle, alpha=0.8)
        ax[0,1].plot((times * code_time_cgs)/ t_linear, v_normalised, color=color, linestyle=linestyle, alpha=0.8)

        




cax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label(r'$d_{\mathrm{sep}}$ [$r_{\mathrm{cl}}$]')
cax.tick_params(axis='y', which='both', color='white', labelcolor='black', direction='in')

# Axis labels
ax[0, 0].set_ylabel(r'$\log\left(m(T < 2T_\mathrm{cl}) / m_0\right)$', labelpad=8)
ax[0, 1].set_ylabel(r'$\Delta v_\mathrm{shear} / v_w$', labelpad = 8)
for axs in [ax[0, 0], ax[0, 1]]:
    axs.set_xlabel(r'$t  [t_{cc}]$')
    

ax[0,0].set_ylim(-3, 1)
ax[0,1].set_ylim(bottom = 0.)


plt.savefig('/u/ferhi/Figures/d_lim.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()
