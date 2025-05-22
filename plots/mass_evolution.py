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

plt.style.use('custom_plot')

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
    
        
def yt_coldgs(run, output_dir=None):
        ds = yt.load(os.path.join(run, 'out'))
        temp = ds.all_data()[('gas', 'temperature')] 
        mass = ds.all_data()[('gas', 'mass')]
        coldg = np.sum(mass[temp <= 2e4])
        ts = ds.current_time
        
        return ts, coldg

def yt_coldgs_hdf(run, N_procs=1):
        if "final" in run:
            return -1, -1
        print(run)
        ds = read_hdf5(run, ['rho', 'T'], n_jobs=N_procs)
        temp = ds['T']
        mass = ds['rho']
        coldg = np.sum(mass[temp <= 2e4])
        ts = float(run.split('.')[-2]) * 0.05/10
        
        return ts, coldg
    
if __name__ == "__main__":
    
    
    plot_yt = False
    plot_hst = True
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
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

    
    #cmap = plt.cm.get_cmap("hsv", len(RUNS))  
    #COLOURS = [cmap(i) for i in range(len(RUNS))]
    COLOURS = [
    'crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen', 
    'red', 'orange',  
    'navy', 'darkgreen', 'firebrick', 'darkorchid', 'darkgoldenrod', 
    'teal', 'indigo', 'tomato', 'peru', 'royalblue'
]

    for j, run in enumerate(run_paths):
        #if "30" in run: continue
        run_name = run  # Get the last part of the path

                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        
        
        tccfact =  depth if sim.tcoolmix/sim.tcc >= 0.1 else 0.1
        tsh = 10 * sim.R_cloud * tccfact / sim.v_wind
        
        #if "fv01_narrow" in run: plot_hst = False; plot_yt = True
        if plot_hst:
            print(run)
            if run in  "/viper/ptmp2/ferhi/d3rcrit/01kc/fv03": continue
            timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run, gout)
            mask = ~np.isnan(norm_mass)
            norm_mass = norm_mass[mask]
            timeseries = timeseries[mask]


            label = run.split('/')[-1]
            plt.style.use('custom_plot')
            plt.plot(timeseries * code_time_cgs / tsh, norm_mass, color=COLOURS[j], label = label)
            if np.sum(cgout) > 10*len(cgout)*1e-22:
                plt.plot(timeseries * code_time_cgs / tsh, cgout, color=COLOURS[j],  alpha = 0.5)
            if np.sum(wgout) > 10*len(cgout)*1e-22:
                plt.plot(timeseries * code_time_cgs / tsh, wgout, color=COLOURS[j], alpha = 0.3)
            if (np.sum(cgout)> 10*len(cgout)*1e-22) & (np.sum(wgout)> 10*len(cgout)*1e-22):
                plt.plot(timeseries * code_time_cgs / tsh, sum, color='black', linestyle='--', alpha = 0.3)
            
        #if "fv01_narrow" in run:
        if plot_yt:
            runs = glob.glob(os.path.join(run, 'out/parthenon.prim.[0-9]*.phdf'))
            print(len(runs))
            initial_mass = None
            for run in runs:
                ts, coldg = yt_coldgs_hdf(run)
            #ts, coldg = run_parallel(runs, func=yt_coldgs_hdf, num_workers=N_procs, output_dir=None)
                label = None
                if initial_mass == None: initial_mass = coldg
                label = run.split('/')[-1] + (' Hst' if 'Hst' in run else '')
                plt.scatter(ts, np.log10(coldg/initial_mass), label=label, color='blue')
                print(f"Cold gas mass: {np.log10(coldg/initial_mass)}")
        plot_hst = True; plot_yt = False
        plt.ylabel(r'$ log(m/m_0)$')
        plt.ylim(bottom=-3, top=1.)


    plt.xlabel(r't [$\tilde t_{cc} = {\scriptstyle \chi^{1/2} L_{ISM} / v_{wind}}$]')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'/u/ferhi/Figures/'+saveFile+'mevol.png')
    plt.show()
