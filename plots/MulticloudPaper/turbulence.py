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

def weighted_percentile(data, percentiles, weights=None):
    """
    Compute weighted percentiles.
    
    Parameters:
        data (array-like): The data.
        percentiles (array-like): Percentiles to compute (0-100).
        weights (array-like): Same shape as `data`. If None, unweighted.
        
    Returns:
        array: Percentile values.
    """
    data = np.asarray(data)
    percentiles = np.asarray(percentiles)
    if weights is None:
        return np.percentile(data, percentiles)
    
    # Sort data and weights
    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]
    
    # Compute the cumulative weight
    cumsum = np.cumsum(weights_sorted)
    cumsum /= cumsum[-1]  # Normalize to 1

    return np.interp(percentiles / 100, cumsum, data_sorted)
    
def hst_turb(run):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))

        mass = data[:,10]
        vt = np.sqrt(data[:,12]*data[:,12] + data[:,14] * data[:, 14])/(mass)
        timeseries = data[:, 0]
        print('vt', vt*code_length_cgs/code_time_cgs)
        
        return timeseries, vt*code_length_cgs/code_time_cgs
    
def hdf_turb(run, mode, cache_file='fix_vturb.npz'):
    runs = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))[:-1]

    # Check if the cache file exists
    if os.path.exists(os.path.join(run, f'{mode}_'+cache_file)):
        print(f"Loading cached data from {mode}_"+cache_file)
        data = np.load(f'{mode}_'+cache_file)
        t, vturb, err_lower, err_upper = data['t'], data['vturb'], data['vturb_err_lower'], data['vturb_err_upper']

        if len(data['t']) == len(runs):
            return t, vturb, err_lower, err_upper
        
        print(f"Cache file {mode}_"+cache_file+" does not match the number of runs. Restarting from last file...")
        t, vturb, err_lower, err_upper = list(t), list(vturb), list(err_lower), list(err_upper)
    else: 
        print(f"Cache file {mode}_"+cache_file+" not found. Starting from scratch...")
        t, vturb, err_lower, err_upper = [], [], [], []


    for file in runs[len(t):70]:
        print("Loading file:", file)
        data = read_hdf5(file, ['rho', 'vel1', 'vel3', 'T'], n_jobs=4)

        if mode == 'cold':
            mask = data["T"] <= 1e5
        elif mode == 'hot':
            mask = data["T"] > 1e5

        # Velocity 1 (x)
        vx = data['vel1'][mask]
        vz = data['vel3'][mask]
        rho = data['rho'][mask]

        vx_var = np.average(vx**2, weights=rho)
        vx_std = np.sqrt(vx_var)

        vz_var = np.average(vz**2, weights=rho)
        vz_std = np.sqrt(vz_var)

        vt = 1.5 ** 0.5 * np.sqrt(vx_std**2 + vz_std**2)

        gen_turb = 1.5 ** 0.5 *np.sqrt(vx**2*rho + vz**2*rho) / np.sqrt(rho)
        err = np.std(gen_turb) 

        # Store time, vturb, and symmetric error bars
        timeseries = float(file.split('/')[-1].split('.')[2])
        t.append(timeseries)
        vturb.append(vt)
        err_lower.append(err)
        err_upper.append(err)

        # Convert to numpy arrays and save
        st = np.array(t)
        svturb = np.array(vturb)
        serr_lower = np.array(err_lower)
        serr_upper = np.array(err_upper)

        np.savez(f'{mode}_'+cache_file, t=st, vturb=svturb, vturb_err_lower=serr_lower, vturb_err_upper=serr_upper)
    print(f"Saved computed data to {mode}_"+cache_file)

    return t, vturb, err_lower, err_upper
    
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
        #if "fv03_long" in run: continue
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

        for mode in ['cold', 'hot']:

            times, v_normalised, err_lower, err_upper = hdf_turb(run, mode=mode)

            # Normalize the velocity and errors
        for mode in ['cold', 'hot']:
            times, v_normalised, err_lower, err_upper = hdf_turb(run, mode=mode)
            v_norm = v_normalised / cs_gas
            err_lower_norm = err_lower / cs_gas
            err_upper_norm = err_upper / cs_gas

            # Plot the central line
            plt.plot(np.array(times) * 0.05, v_norm, label=mode, color=COLOURS[j])

            # Fill between error bars
            plt.fill_between(times * 0.05,
                            v_norm - err_lower_norm,
                            v_norm + err_upper_norm,
                            color=COLOURS[j],
                            alpha=0.3)
            
            
            plt.ylabel(r'$ v_{turb} / c_{\mathrm{s, cold}}$')
            #plt.ylim(top=1.2, bottom = 0)


            print(saveFile)
            plt.xlabel(r't ')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f'/u/ferhi/Figures/'+saveFile+'_'+mode+'vturb.png')
            plt.show()

