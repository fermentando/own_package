import numpy as np
import matplotlib.pyplot as plt
from read_hdf5 import read_hdf5
import os
import glob
from matplotlib.colors import LogNorm
from adjust_ics import *
import seaborn as sns

cm = sns.color_palette("blend:#7AB,#EDA", as_cmap=True)

def T_v_phase(run, vol,v_wind, v_correction=None, output_dir=None):
    data = read_hdf5(run, ['rho', 'vel2', 'T'], n_jobs=4)
    T_flat = data['T'].flatten()
    vy_flat = data['vel2'].flatten() + v_correction
    rho_flat = data['rho'].flatten()

    if v_correction is None:
        v_correction = 0

    mask = (T_flat > 0) & (np.abs(vy_flat) > 0)
    T_log = T_flat[mask]
    vy_log = np.abs(vy_flat[mask]) / v_wind
    rho_log = rho_flat[mask]

    T_bins = np.logspace(3, 7, 100)
    vy_bins = np.logspace(-6, 1, 100)

    plt.figure(figsize=(8, 6))
    plt.style.use('custom_plot')
    plt.hist2d(vy_log, T_log, bins=[vy_bins, T_bins], weights=rho_log * vol / 1.989e33,
            cmap=cm, norm=LogNorm(vmin=1e-1, vmax=1e5))

    plt.xscale('log')
    plt.yscale('log')

    plt.colorbar(label=r'Total Mass $M_\odot$')
    plt.ylabel('T [K]')
    plt.xlabel(r'$v_{\mathrm{gas}} / v_w$')
    plt.xlim(1e-6,10)
    plt.ylim(1e3, 1e7)
    plt.tight_layout()
    t_indx = float(run.split('/')[-1].split('.')[2])
    plt.savefig(f'/u/ferhi/Figures/T_v_plots/{output_dir}/{t_indx}_T_v_phase.png')

def T_v_phase_multi(run_list, vol, mass_cloud, v_wind, v_correction_list=None, output_dir=None):
    if v_correction_list is None:
        v_correction_list = [0.0] * len(run_list)

    plt.style.use('custom_plot')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)


    T_bins = np.logspace(4, 7, 80)
    vy_bins = np.logspace(-5, 1, 80)

    for i, (run, v_correction) in enumerate(zip(run_list, v_correction_list)):
        print(f"Processing figure {i}...")
        data = read_hdf5(run, ['rho', 'vel2', 'T'], n_jobs=4)
        T_flat = data['T'].flatten()
        vy_flat = data['vel2'].flatten() + v_correction
        vy_flat[vy_flat < 1e-5] = 2e-5
        rho_flat = data['rho'].flatten()

        mask = (T_flat > 0) & (np.abs(vy_flat) > 0)
        T_log = T_flat[mask]
        vy_log = np.abs(vy_flat[mask]) / v_wind
        rho_log = rho_flat[mask]
        plt.style.use('custom_plot')

        h = axes[i].hist2d(vy_log, T_log, bins=[vy_bins, T_bins],
                           weights=rho_log * vol / mass_cloud,
                           cmap=cm, norm=LogNorm(vmin=1e-2, vmax =1e2))

        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(1e-5, 10)
        axes[i].set_ylim(9e3, 1e7)
        axes[i].set_xlabel(r'$v_{\mathrm{gas}} / v_w$')
        #axes[i].axvline(x=1, color='k', linestyle='--', alpha = 0.5)


    axes[0].set_ylabel('T [K]')

    # Add colorbar to the right of all subplots
    cbar_ax = fig.add_axes([0.92, 0.20, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(h[3], cax=cbar_ax, label=r'$m_\mathrm{tot} [m_\mathrm{clump}$]')

    fig.tight_layout(rect=[0, 0, 0.92, 1])  # leave space on right for colorbar
    save_path = f'/u/ferhi/Figures/T_v_plots/{output_dir}/T_v_phase_multi.png'
    print("File saved in : ", save_path)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == '__main__':

    RUNS = [os.getcwd()]
    run_paths = ["/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/"]
    parts = run_paths[0].split('/')
    saveFile = run_paths[0].split('ferhi/')[-1]

    if not os.path.exists(os.path.join('/u/ferhi/Figures/T_v_plots/',saveFile)): 
        os.makedirs(os.path.join('/u/ferhi/Figures/T_v_plots/',saveFile))

    sim = SingleCloudCC(os.path.join(run_paths[0], 'ism.in'), dir=run_paths[0])
    code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
    code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))

    output_delay = int(float(sim.reader.get('parthenon/output0', 'dt'))/ float(sim.reader.get('parthenon/output1', 'dt')))
    v_boost = np.loadtxt(os.path.join(run_paths[0], 'out/parthenon.out1.hst'))[:, -1] 

    files = np.sort(glob.glob(os.path.join(run_paths[0], 'out/parthenon.prim.*.phdf')))[1:-1]
    indexes = output_delay * np.asarray(range(1,len(files)))

    v_correction = v_boost[indexes]
    
    rho_cloud_cgs = float(sim.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
    m_cloud = 4 * np.pi / 3 * sim.R_cloud**3 * rho_cloud_cgs
    
    run_list_multiplot = [
        "/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00001.phdf", 
        "/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00018.phdf",
        "/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/out/parthenon.prim.00100.phdf", 
    ]
    
    v_correction_list_multiplot =  np.asarray([1,18,100])
    
    T_v_phase_multi(run_list_multiplot, vol =code_length_cgs**3, mass_cloud = m_cloud, v_correction_list = v_correction_list_multiplot, v_wind = sim.v_wind, output_dir = saveFile )

    #for j, file in enumerate(files):
    #    print('Processing file:', file)
    #    v_correction_single = v_correction[j] * code_length_cgs / code_time_cgs
    #    T_v_phase(file, vol = code_length_cgs**3, v_correction=v_correction_single, v_wind = sim.v_wind,output_dir = saveFile)

