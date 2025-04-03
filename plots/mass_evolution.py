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

plt.style.use('custom_plot')

def run_parallel(runs, func, num_workers=2,  *args):
    
    with Pool(num_workers) as pool:
        results = pool.starmap(func, [(run, *args) for run in runs])
    
    num_outputs = len(results[0]) 
    output_arrays = [[] for _ in range(num_outputs)]
    
    # Distribute results into the output arrays
    for res in results:
        for i in range(num_outputs):
            output_arrays[i].append(res[i])
    
    output_arrays = [np.array(arr) for arr in output_arrays]
    
    return output_arrays

def get_n_procs():
    parser = argparse.ArgumentParser(description="Set the number of processors.")
    parser.add_argument("N_procs", nargs="?", type=int, default=1, help="Number of processors to use.")
    args = parser.parse_args()
    return max(1, min(args.N_procs, cpu_count()))  # Ensure valid range

def hst_evolution(run):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        data = np.where(data==0, 1e-22, data)
        norm_mass = np.log10(data[:, 10]/data[0, 10])
        timeseries = data[:, 0]

        return timeseries, norm_mass
        
def yt_coldgs(run):
        ds = yt.load(run)
        temp = ds.all_data()[('gas', 'temperature')] 
        mass = ds.all_data()[('gas', 'mass')]
        coldg = np.sum(mass[temp <= 2e4])
        ts = ds.current_time
        
        return ts, coldg

    
if __name__ == "__main__":
    
    
    plot_yt = False
    plot_hst = True
    
    RUNS = [os.getcwd()]
    print(RUNS)
    parts = RUNS[0].split('/')
    saveFile = f"{parts[-2]}/{parts[-1]}"
    if not os.path.exists(os.path.join('/u/ferhi/Figures/',parts[-2])): 
        os.makedirs(os.path.join('/u/ferhi/Figures/',parts[-2]))
    
    N_procs = get_n_procs()
    print(f"N_procs set to: {N_procs} processors.")


    # Execution
    #Set to True if you would like to analyse runs without coldg mass Hst output
    NonHistFiles = False
    run_paths = RUNS
    #run_paths = np.array([os.path.join(runDir, run) for run in RUNS])
    if False:
        run_paths = np.array([
            os.path.join(runDir, folder) 
            for folder in os.listdir(runDir) 
            if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder)) 
        ])
    RUNS = np.append(run_paths, np.array([run for run in glob.glob(os.path.join(runDir, 'HstCons', 'fv*'))])) if NonHistFiles else run_paths


    #cmap = plt.cm.get_cmap("hsv", len(RUNS))  
    #COLOURS = [cmap(i) for i in range(len(RUNS))]
    COLOURS = [
    'crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen', 
    'red', 'orange',  
    'navy', 'darkgreen', 'firebrick', 'darkorchid', 'darkgoldenrod', 
    'teal', 'indigo', 'tomato', 'peru', 'royalblue'
]

    for j, run in enumerate(RUNS):
        run_name = run  # Get the last part of the path
        print(run)
                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        
        tccfact = float(sim.reader.get('problem/wtopenrun', 'depth')) if sim.tcoolmix/sim.tcc <= 1 else 0.1
        print(tccfact)
        

        if plot_hst:
            try:
                timeseries, norm_mass = hst_evolution(run)
                norm_mass = norm_mass[~np.isnan(norm_mass)]
                timeseries = timeseries[~np.isnan(norm_mass)]
                print(norm_mass)
            except:
                continue
            label = run.split('/')[-1]
            plt.plot(timeseries/sim.tcc * code_time_cgs / tccfact, norm_mass, color=COLOURS[j], label = label)

        if plot_yt:
            ts, coldg = run_parallel(files, func=yt_coldgs, num_workers=N_procs)
            label = None
            initial_mass = coldg[0]
            label = run.split('/')[-1] + (' Hst' if 'Hst' in run else '')
            plt.scatter(ts/sim.tcc * code_time_cgs / tccfact, np.log10(coldg/initial_mass), label=label, color=COLOURS[j])
            
        plt.ylabel(r'$ log(m/m_0)$')
        plt.ylim(bottom=-3)
        saveFile +='massplot'


    plt.xlabel(r't [$t_{cc, eff}$]')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'/u/ferhi/Figures/'+saveFile+'.png')
    plt.show()
