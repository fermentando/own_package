import os
import yt 
import glob
import sys
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt

plt.style.use('custom_plot')


def hst_evolution(run):
        data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
        data = np.where(data==0, 1e-22, data)
        norm_mass = np.log10(data[:, -1]/data[0, -1])
        timeseries = data[:, 0]

        return timeseries, norm_mass
        
def yt_coldgs(run):
        ds = yt.load(run)
        temp = ds.all_data()[('gas', 'temperature')] 
        mass = ds.all_data()[('gas', 'mass')]
        coldg = np.sum(mass[temp <= 2e4])
        ts = ds.current_time
        
        return ts, coldg
    
def yt_entrainment(run, wind = False):
    
        ds = yt.load(run)
        temp = ds.all_data()[('gas', 'temperature')] 
        vels_i = ds.all_data()[('gas', 'velocity_y')]
        velg = np.mean(vels_i[temp <= 1e5])        
        velw = np.mean(vels_i[temp >=1e6])   
        
        ts = ds.current_time
                
        return ts, velg, velw
    
if __name__ == "__main__":
    
    plot_mass = True
    
    plot_norm_v = False
    
    #Directories
    saveFile = 'ISM_slab/kc_thinslab_highres'
    runDir = '/raven/ptmp/ferhi/ISM_thinslab/'
    RUNS = ['fv01e','fv01_highres/']

    #Set to True if you would like to analyse runs without coldg mass Hst output
    NonHistFiles = False
    run_paths = np.array([os.path.join(runDir, run) for run in RUNS])
    RUNS = np.append(run_paths, np.array([run for run in glob.glob(os.path.join(runDir, 'HstCons', 'fv*'))])) if NonHistFiles else run_paths


    #cmap = plt.cm.get_cmap("hsv", len(RUNS))  
    #COLOURS = [cmap(i) for i in range(len(RUNS))]
    COLOURS = ['crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen', 'red', 'orange']

    for j, run in enumerate(RUNS):
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        
        if plot_mass:
        
            if True:
                timeseries, norm_mass = hst_evolution(run)
                label = run.split('/')[-1]
                print(sim.tcoolmix / sim.tcc)
                plt.plot(timeseries/sim.tcc * code_time_cgs, norm_mass, color=COLOURS[j], label = label)

            if False:
                for filename in files:
                    ts, coldg = yt_coldgs(filename)
                    label = None
                    if filename == files[0]:
                        initial_mass = coldg
                        label = run.split('/')[-1] + (' Hst' if 'Hst' in run else '')
                    plt.scatter(ts/sim.tcc * code_time_cgs, np.log10(coldg/initial_mass), label=label, color=COLOURS[j])
        if plot_norm_v:
            
            times = []
            v_normalised = []
            
            for filename in files:
                    ts, vg, vw = yt_entrainment(filename)
                    times.append(ts/sim.tcc * code_time_cgs)
                    v_normalised.append(1 - np.float64(vg/vw))

            plt.plot(times, v_normalised,  label=run.split('/')[-1], color=COLOURS[j])



    plt.xlabel(r't [$t_{cc}$]')
    plt.ylabel(r'$ log(m/m_0)$')
    plt.legend(loc='lower right')
    #plt.xlim(right=1)
    if plot_norm_v:
        plt.ylim(top=1.2, bottom = 0)
    if plot_mass:
        plt.ylim(bottom=-3)
    plt.tight_layout()
    plt.savefig(f'/u/ferhi/Figures/'+saveFile+'.png')
    plt.show()
