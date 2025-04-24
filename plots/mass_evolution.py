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

def hst_evolution(run, gout=False):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        data = np.where(data==0, 1e-22, data)
        norm_mass = np.log10(data[:, 10]/data[0, 10])
        timeseries = data[:, 0]
        
        wgout = np.zeros_like(timeseries); cgout = wgout
        sum = norm_mass
        
        if gout: 
            wgout = np.log10(data[:, -2]/data[0, 10]) 
            cgout = np.log10(data[:, -3]/data[0, 10])
            sum = np.log10((data[:, 10]+data[:, -2]+data[:, -3])/data[0,10])
        return timeseries, norm_mass, cgout, wgout, sum
    
        
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
    
    user_args = get_user_args(sys.argv)
    gout = True
    
    if len(user_args) > 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
        print('Saved to: ', saveFile)
        if not os.path.exists(os.path.join('/u/ferhi/Figures/',f"{parts[-3]}/{parts[-2]}")): 
            os.makedirs(os.path.join('/u/ferhi/Figures/',f"{parts[-3]}/{parts[-2]}"))

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



    
    N_procs = get_n_procs()
    print(f"N_procs set to: {N_procs} processors.")
    
    #cmap = plt.cm.get_cmap("hsv", len(RUNS))  
    #COLOURS = [cmap(i) for i in range(len(RUNS))]
    COLOURS = [
    'crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen', 
    'red', 'orange',  
    'navy', 'darkgreen', 'firebrick', 'darkorchid', 'darkgoldenrod', 
    'teal', 'indigo', 'tomato', 'peru', 'royalblue'
]

    for j, run in enumerate(run_paths):
        run_name = run  # Get the last part of the path

                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        print(sim.tcoolmix/sim.tcc)
        
        tccfact = float(sim.reader.get('problem/wtopenrun', 'depth')) if sim.tcoolmix/sim.tcc >= 0.1 else 0.1
        print(tccfact)
        

        if plot_hst:

            try:
                timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run, gout)
                norm_mass = norm_mass[~np.isnan(norm_mass)]
                timeseries = timeseries[~np.isnan(norm_mass)]
            except:
                continue

            label = run.split('/')[-1]
            plt.plot(timeseries/sim.tcc * code_time_cgs / tccfact, norm_mass, color=COLOURS[j], label = label)
            if np.sum(cgout) > 10*len(cgout)*1e-22:
                plt.plot(timeseries/sim.tcc * code_time_cgs / tccfact, cgout, color=COLOURS[j],  alpha = 0.5)
            if np.sum(wgout) > 10*len(cgout)*1e-22:
                plt.plot(timeseries/sim.tcc * code_time_cgs / tccfact, wgout, color=COLOURS[j], alpha = 0.3)
            if (np.sum(cgout)> 10*len(cgout)*1e-22) & (np.sum(wgout)> 10*len(cgout)*1e-22):
                plt.plot(timeseries/sim.tcc * code_time_cgs / tccfact, sum, color='black', linestyle='--', alpha = 0.3)
            

        if plot_yt:
            ts, coldg = run_parallel(files, func=yt_coldgs, num_workers=N_procs)
            label = None
            initial_mass = coldg[0]
            label = run.split('/')[-1] + (' Hst' if 'Hst' in run else '')
            plt.scatter(ts/sim.tcc * code_time_cgs / tccfact, np.log10(coldg/initial_mass), label=label, color=COLOURS[j])
            
        plt.ylabel(r'$ log(m/m_0)$')
        plt.ylim(bottom=-3)


    print(saveFile)
    plt.xlabel(r't [$t_{cc, eff}$]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'/u/ferhi/Figures/'+saveFile+'mevol.png')
    plt.show()
