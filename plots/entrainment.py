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
    
def hst_entrainment(run, vwind):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        vboost = data[:, -1]
        if np.shape(data)[1] >= 17:
            mass = data[:,11]
            vx2 = abs(data[:,13])/(mass)
        else: 
            mass = data[:,10]
            vx2 = abs(data[:,12])/(mass)
        delta_v = (vwind - (vx2 + vboost))/vwind
        print(vwind)
        print(delta_v)
        timeseries = data[:, 0]
        
        return timeseries, delta_v
    
def yt_entrainment(run, wind = False):
        
        ds = yt.load(run)
        temp = ds.all_data()[('gas', 'temperature')] 
        mass = ds.all_data()[('gas', 'mass')]
        vels_i = ds.all_data()[('gas', 'velocity_y')].to('km/s')
        mask_hot = (temp >= 1e6)  &  (vels_i > 2)
        mask_cold = temp <= 5e4
        velg = np.average(vels_i[mask_cold], weights=mass[mask_cold]) if mask_cold.any() else np.nan
        velw = np.average(vels_i[mask_hot], weights=mass[mask_hot]) if mask_hot.any() else np.nan
        #check number cells
        print('This is vel coldg:', velg)
        print('And this is vel hotg: ', velw)
        
        
        ts = ds.current_time
                
        return ts, velg, velw
    
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
        print(run)
                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        print(sim.tcoolmix/sim.tcc)
        
        tccfact = float(sim.reader.get('problem/wtopenrun', 'depth')) if sim.tcoolmix/sim.tcc >= 0.1 else 0.1
        print(tccfact)


        
        if plot_hst: 
            plt.style.use('custom_plot')
            if run == "/viper/ptmp2/ferhi/d3rcrit/01kc/fv03": continue
            code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
            code_mass_cgs = float(sim.reader.get('units', 'code_mass_cgs'))
            v_wind = sim.v_wind / code_length_cgs * code_time_cgs
            times, v_normalised = hst_entrainment(run, 110)
            plt.plot(times/sim.tcc * code_time_cgs / tccfact, v_normalised,  label=run.split('/')[-1], color=COLOURS[j])
            
        if plot_yt:               


            times, vg, vw = run_parallel(files, func=yt_entrainment, num_workers=N_procs)
            v_normalised = (vw - vg)/110
            print('This is normalised v: ', v_normalised)

            plt.scatter(times/sim.tcc * code_time_cgs / tccfact, v_normalised,  label=run.split('/')[-1], color=COLOURS[j])
            
        plt.ylabel(r'$ \Delta_v /v_0$')
        plt.ylim(top=1.2, bottom = 0)



    plt.xlabel(r't [$t_{cc, eff}$]')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'/u/ferhi/Figures/'+saveFile+'vevol.png')
    plt.show()
