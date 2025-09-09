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
    
def hdf_column_density(run, mode, dy, cache_file='column_density.npz'):

    runs = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))[:-1]

    cache_path = os.path.join(run, f'{mode}_' + cache_file)
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        t, col_dens, err_lower, err_upper = data['t'], data['col_dens'], data['err_lower'], data['err_upper']
        if len(t) == len(runs):
            return t, col_dens, err_lower, err_upper
        print("Cache incomplete. Resuming...")
        t, col_dens, err_lower, err_upper = list(t), list(col_dens), list(err_lower), list(err_upper)
    else:
        print(f"Starting fresh for mode '{mode}'...")
        t, col_dens, err_lower, err_upper = [], [], [], []

    for file in runs[len(t):]:
        print("Reading:", file)
        data = read_hdf5(file, ['rho', 'T'], n_jobs=4)
        rho = data['rho']
        temp = data['T']

        if mode == 'cold':
            mask = temp <= 1e5
        elif mode == 'hot':
            mask = temp > 1e5
        else:
            mask = np.ones_like(temp, dtype=bool)  # no filtering

        rho_masked = np.where(mask, rho, 0.0)

        # Column density along y-axis
        sigma_y = np.sum(rho_masked, axis=1) * dy  # shape (Nx, Nz)

        # Average and error
        col_avg = np.mean(sigma_y)
        col_std = np.std(sigma_y)

        time = float(file.split('/')[-1].split('.')[2])
        t.append(time)
        col_dens.append(col_avg)
        err_lower.append(col_std)
        err_upper.append(col_std)

        np.savez(cache_path,
                 t=np.array(t),
                 col_dens=np.array(col_dens),
                 err_lower=np.array(err_lower),
                 err_upper=np.array(err_upper))

    print(f"Saved column density data to {cache_path}")
    return t, col_dens, err_lower, err_upper

    
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
        dy = (float(sim.reader.get('parthenon/mesh', 'x2max')) - float(sim.reader.get('parthenon/mesh', 'x2min')))/ float(sim.reader.get('parthenon/mesh', 'nx2'))
        dy *= float(sim.reader.get('units', 'code_length_cgs')) 
        


        

        plt.style.use('custom_plot')


        for mode in ['cold']:
            times, col_dens, err_lower, err_upper = hdf_column_density(run, mode=mode, dy = dy)


            # Plot the central line
            plt.plot(np.array(times) * 0.05, col_dens, label=mode, color=COLOURS[j])

            # Fill between error bars
            plt.fill_between(times * 0.05,
                            col_dens - err_lower,
                            col_dens + err_upper,
                            color=COLOURS[j],
                            alpha=0.3)
            
            
            plt.ylabel(r'$ N_\mathrm{HII}$')
            plt.yscale('log')
            #plt.ylim(top=1.2, bottom = 0)


            print(saveFile)
            plt.xlabel(r't ')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f'/u/ferhi/Figures/'+saveFile+'_'+mode+'col_dens.png')
            plt.show()

