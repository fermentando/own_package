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
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


# Set up a colormap using seaborn
cmap = sns.color_palette("mako", as_cmap=True)  # or "magma", "plasma", etc.
norm = mcolors.LogNorm(vmin=1, vmax=1000)  # Log scale if range is wide
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


#plt.style.use('custom_plot')
linestyles = {1:'-', 0:'--', 3:'-.', 2:':'}

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
    problem_name = 'stratified_box'

    fig, ax = plt.subplots(figsize=(8, 6))
    
    N_procs, user_args = get_n_procs_and_user_args()

    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    if len(user_args) > 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-1]}"
        print('Saved to: ', saveFile)


    
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
        if "fv01_copy" in run: continue
        if "fv02_r1e3" in run: continue
        if "2x" in run: continue

        if "4x" in run: continue

                
        sim = SingleCloudCC(os.path.join(run, 'strat.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        depth = float(sim.reader.get(f'problem/{problem_name}', 'depth'))
        rho_wind = float(sim.reader.get(f'problem/{problem_name}', 'rho_wind_cgs'))
        rho_cloud = float(sim.reader.get(f'problem/{problem_name}', 'rho_cloud_cgs'))
        chi = rho_cloud / rho_wind
        dt = float(sim.reader.get('parthenon/output1', 'dt'))
        dt2 = float(sim.reader.get('parthenon/output0', 'dt'))
            
        
        try:
            fv = float(sim.reader.get(f'problem/{problem_name}', 'fv'))
            base_fv = int(-np.log10(fv))
        except:
            base_fv = int(run.split('fv')[-1][:2])
            fv = 10 ** (-base_fv)


        tsh =  depth * sim.R_cloud / sim.v_wind

        t1 = np.sqrt(chi) * 0.1 * sim.R_cloud / sim.v_wind
        t2 = np.sqrt(chi) * fv * depth *  sim.R_cloud / sim.v_wind
        print("This is compression lim:", (sim.tcoolmix * sim.v_wind )/ 0.1 / sim.R_cloud)

        # Linear sum
        t_linear =  t2 
        
        
        tccfact =  depth if sim.tcoolmix/sim.tcc >= 0.1 else 0.1

        
        #if "fv01_narrow" in run: plot_hst = False; plot_yt = True
        #if "v02" in run: continue
        if plot_hst:
            print(run)
            #if run in  "/viper/ptmp2/ferhi/d3rcrit/01kc/fv03": continue
            try:
                timeseries, norm_mass, cgout, wgout, sum = hst_evolution(run, gout)
            except Exception as e:
                 continue
            mask = ~np.isnan(norm_mass)
            norm_mass = norm_mass[mask]
            timeseries = timeseries[mask]
            color = sm.to_rgba(tccfact)

            time_axis = timeseries * code_time_cgs 
            idx = (np.abs(time_axis - tsh)).argmin()
            t_shift = 0.65 * tsh if j ==1 else 0
            correction_index_mass = (np.abs(time_axis - t_shift)).argmin()


            label = run.split('/')[-1]
            plt.style.use('custom_plot')
            ax.plot((timeseries * code_time_cgs - t_shift) / t_linear, norm_mass-norm_mass[correction_index_mass], color=color, linestyle=linestyles[base_fv], alpha=0.8)
            if np.sum(cgout) > 10*len(cgout)*1e-22:
                ax.plot(timeseries * code_time_cgs / tsh, cgout, color=COLOURS[j],  alpha = 0.5)
            if np.sum(wgout) > 10*len(cgout)*1e-22:
                ax.plot(timeseries * code_time_cgs / tsh, wgout, color=COLOURS[j], alpha = 0.3)
            if (np.sum(cgout)> 10*len(cgout)*1e-22) & (np.sum(wgout)> 10*len(cgout)*1e-22):
                ax.plot(timeseries * code_time_cgs / tsh, sum, color='black', linestyle='--', alpha = 0.3)
            
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
    linestyles = {1:'-', 0:'--', 3:'-.', 2:':'}

    fv_legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label=r'$f_v = 10^{\mathrm{-1}}$'),
        Line2D([0], [0], color='black', linestyle=':', label=r'$f_v = 10^{\mathrm{-2}}$'),
        Line2D([0], [0], color='black', linestyle='-.', label=r'$f_v = 10^{\mathrm{-3}}$'),
    ]

    # Add to plot
    legend1 = ax.legend(
        handles=fv_legend_elements,
        loc='upper center',
        ncol=3,
    )
    ax.add_artist(legend1) 
    ax.set_ylabel(r'$log(m/m_0)$')
    ax.set_xlabel(r't [$\tilde t_{cc} $]')
    ax.set_ylim(bottom=-3, top=1.)

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r'$L_{\mathrm{ISM}} [r_{\mathrm{cloud}}]$')

    # Save and show
    print(f"Saved to: {saveFile}mevol.png")
    plt.tight_layout()
    plt.savefig(f'{saveFile}mevol.png')
    plt.show()


