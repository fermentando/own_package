import os
import yt 
import glob
import sys
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from read_hdf5 import read_hdf5
import argparse

#plt.style.use('custom_plot')
COLOURS = [
'crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen', 
'red', 'orange',  
'navy', 'darkgreen', 'firebrick', 'darkorchid', 'darkgoldenrod', 
'teal', 'indigo', 'tomato', 'peru', 'royalblue'
]
    

def plot_turbulent_velocity_histograms_vs_y(run, snapshot_index=-1, wind = 1,  mode='cold', smooth_window=1):

    runs = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
    file = runs[snapshot_index]
    print("Loading file:", file)

    data = read_hdf5(file, ['rho', 'vel1', 'vel3', 'T'], n_jobs=12)

    rho = data['rho']
    vx = data['vel1']
    vz = data['vel3']
    T = data['T']

    if mode == 'cold':
        mask = T <= 1e5
    elif mode == 'hot':
        mask = T > 1e5
    else:
        raise ValueError("mode must be 'cold' or 'hot'")

    ny = rho.shape[1]
    y_coords = np.arange(ny)  # Replace with actual coordinates if needed
    vx_std_y, vz_std_y = [], []

    for j in range(ny):
        slice_mask = mask[:, j, :]
        if not np.any(slice_mask):
            vx_std_y.append(0)
            vz_std_y.append(0)
            continue

        rho_slice = rho[:, j, :][slice_mask]
        vx_slice = vx[:, j, :][slice_mask]
        vz_slice = vz[:, j, :][slice_mask]

        vx_avg = np.average(vx_slice, weights=rho_slice)
        vx_var = np.average((vx_slice - vx_avg) ** 2, weights=rho_slice)
        vx_std_y.append(np.sqrt(vx_var))

        vz_avg = np.average(vz_slice, weights=rho_slice)
        vz_var = np.average((vz_slice - vz_avg) ** 2, weights=rho_slice)
        vz_std_y.append(np.sqrt(vz_var))

    vx_std_y = np.array(vx_std_y)
    vz_std_y = np.array(vz_std_y)

    # Optional smoothing
    def smooth(arr, window):
        if window <= 1:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')

    vx_smooth = smooth(vx_std_y, smooth_window)
    vz_smooth = smooth(vz_std_y, smooth_window)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axs[0].scatter(y_coords, vx_smooth/wind, label=r'$v_x$', color='tab:blue')
    axs[0].set_xlabel('y index')
    axs[0].set_ylabel(r'Turbulent velocity')
    axs[0].set_title(r'$\rho$-weighted $v_x$ vs $y$')

    axs[1].scatter(y_coords, vz_smooth/wind, label=r'$v_z$', color='tab:orange')
    axs[1].set_xlabel('y index')
    axs[1].set_title(r'$\rho$-weighted $v_z$ vs $y$')

    fig.suptitle(f'Turbulent Velocity Distributions ({mode})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'turbulent_hists.png')

    plt.show()

if __name__ == "__main__":
    

    plot_yt = True
    plot_hst = False
    mode = 'cold'  # or 'hot'
    
    user_args = get_user_args(sys.argv)
    
    if len(user_args) > 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
        print('Saved to: ', saveFile)
        if not os.path.exists(os.path.join('/u/ferhi/Figures/',saveFile)): 
            os.makedirs(os.path.join('/u/ferhi/Figures/',saveFile))

    #run_paths = np.array([os.path.join(runDir, run) for run in RUNS])
    else:
        runDir = os.getcwd()
        run_paths = np.array([
            os.path.join(runDir, folder) 
            for folder in os.listdir(runDir) 
            if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder)) 
        ])
        parts = runDir.split('/')
        saveFile = f"{parts[-2]}/{parts[-1]}"
        if not os.path.exists(os.path.join('/u/ferhi/Figures/',parts[-2])): 
            os.makedirs(os.path.join('/u/ferhi/Figures/',parts[-2]))


    
    N_procs = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    print(f"RUNS: {run_paths}")
    

    COLOURS = [
    'crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen', 
    'red', 'orange',  
    'navy', 'darkgreen', 'firebrick', 'darkorchid', 'darkgoldenrod', 
    'teal', 'indigo', 'tomato', 'peru', 'royalblue'
]

    for j, run in enumerate(run_paths):
        run_name = run  # Get the last part of the path
        if "fv03_long" in run: continue
        print(run)
                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        
        tccfact =  depth if sim.tcoolmix/sim.tcc >= 0.1 else 0.1
        tsh = 10 * sim.R_cloud * tccfact / sim.v_wind
        cs_gas = np.sqrt(5/3 * ut.constants.kb * sim.T_cloud / sim.mbar)


        

        plt.style.use('custom_plot')
        #if run == "/viper/ptmp2/ferhi/d3rcrit/01kc/fv03": continue

        plot_turbulent_velocity_histograms_vs_y(run, mode=mode, wind = cs_gas, snapshot_index=30)

    
