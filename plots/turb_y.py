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
    
def hst_turb(run):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))

        mass = data[:,10]
        vt = np.sqrt(data[:,12]*data[:,12] + data[:,14] * data[:, 14])/(mass)
        timeseries = data[:, 0]
        print('vt', vt*code_length_cgs/code_time_cgs)
        
        return timeseries, vt*code_length_cgs/code_time_cgs

def rolling_average(y_vals, vturb_y, window_size=10):
    kernel = np.ones(window_size) / window_size
    vturb_smooth = np.convolve(vturb_y, kernel, mode='same')
    return y_vals, vturb_smooth

def hdf_turb_vs_y(run, snapshot_index=-1, mode='cold', cache_file='vturb_vs_y_cache.npz'):
    cache_path = f'{snapshot_index}_{mode}_{cache_file}'
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        y, v =rolling_average(data['y'], data['vturb_y'], window_size=1)
        return y, v

    # Load a single snapshot
    runs = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
    file = runs[snapshot_index]
    print("Loading file:", file)

    data = read_hdf5(file, ['rho', 'vel1', 'vel3', 'T'], n_jobs=12)

    # Assume data has dimensions (nx, ny, nz)
    # Axis 1 is y-direction
    rho = data['rho']
    vel1 = data['vel1']
    vel3 = data['vel3']
    T = data['T']

    if mode == 'cold':
        mask = T <= 1e5
    elif mode == 'hot':
        mask = T > 1e5
    else:
        raise ValueError("mode must be 'cold' or 'hot'")

    ny = rho.shape[1]
    vturb_y = []
    y_coords = np.arange(ny//1)  # Replace with real y-coordinates if available

    for j in range(ny//1):
        slice_mask = mask[:, 1*j, :]
        if not np.any(slice_mask):
            vturb_y.append(0)
            continue

        rho_slice = rho[:, 1*j, :][slice_mask]
        vx_slice = vel1[:, 1*j, :][slice_mask]
        vz_slice = vel3[:, 1*j, :][slice_mask]

        vx_var = np.average(vx_slice**2, weights=rho_slice)
        vx_std = np.sqrt(vx_var)

        vz_var = np.average(vz_slice**2, weights=rho_slice)
        vz_std = np.sqrt(vz_var)

        vt = np.sqrt(vx_std**2 + vz_std**2)
        vturb_y.append(vt)

    vturb_y = np.array(vturb_y)
    #np.savez(cache_path, y=y_coords, vturb_y=vturb_y)
    #print(f"Saved turbulent velocity profile to {cache_path}")
    
    y_coords, vturb_y = rolling_average(y_coords, vturb_y, window_size=30)
    return y_coords, vturb_y


if __name__ == "__main__":
    

    plot_yt = True
    plot_hst = False
    mode = 'hot'  # or 'hot'
    
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


        

        plt.style.use('custom_plot')
        #if run == "/viper/ptmp2/ferhi/d3rcrit/01kc/fv03": continue
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        code_mass_cgs = float(sim.reader.get('units', 'code_mass_cgs'))
        v_wind = sim.v_wind / code_length_cgs * code_time_cgs
        cs_gas = np.sqrt(5/3 * ut.constants.kb * sim.T_cloud / sim.mbar)
        print('cs_gas', cs_gas)

        ycords, v_normalised = hdf_turb_vs_y(run, mode=mode, snapshot_index=30)
        print('v_normalised', v_normalised/cs_gas)
        plt.plot(ycords, v_normalised / cs_gas,  label=run.split('/')[-1], color=COLOURS[j])
        
        
        plt.ylabel(r'$ v_{turb} / c_{\mathrm{s, cold}}$')
        #plt.ylim(top=1.2, bottom = 0)


        print(saveFile)
        plt.xlabel(r'y')
        plt.tight_layout()
        plt.savefig(f'/u/ferhi/Figures/'+saveFile+'_corrected_std_'+mode+'vturb.png')
        plt.show()

