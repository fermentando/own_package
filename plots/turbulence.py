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

#plt.style.use('custom_plot')
    
def hst_turb(run):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))

        mass = data[:,10]
        vt = np.sqrt(data[:,12]*data[:,12] + data[:,14] * data[:, 14])/(mass)
        timeseries = data[:, 0]
        
        return timeseries, vt
    

    
if __name__ == "__main__":
    

    plot_yt = False
    plot_hst = True
    
    user_args = get_user_args(sys.argv)
    
    if user_args:
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
        if "fv02" in run: continue
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
        times, v_normalised = hst_turb(run)
        plt.plot(times * code_time_cgs / tsh, v_normalised / sim.v_wind,  label=run.split('/')[-1], color=COLOURS[j])
        
        
        plt.ylabel(r'$ v_{turb} [kms^{-1}$')
        #plt.ylim(top=1.2, bottom = 0)



    plt.xlabel(r't ')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'/u/ferhi/Figures/'+saveFile+'vturb.png')
    plt.show()
