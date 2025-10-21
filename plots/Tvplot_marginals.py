import numpy as np
import matplotlib.pyplot as plt
from read_hdf5 import read_hdf5
import os
import glob
from matplotlib.colors import LogNorm
from adjust_ics import *
import seaborn as sns
from scipy.optimize import curve_fit

cm = sns.color_palette("blend:#7AB,#EDA", as_cmap=True)

def y(x, A, alpha, xc, beta):
    return A * x**alpha * np.exp(-(x/xc)**beta)

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
            cmap=cm, norm=LogNorm(vmin=1e1, vmax=1e5))

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

    # Create 2 rows (top for curve, bottom for 2D hist), shared x
    fig, axes = plt.subplots(2, len(run_list), 
                             figsize=(5*len(run_list), 6), 
                             sharex='col', 
                             gridspec_kw={'height_ratios': [1, 4]})
    
    # bins for hist
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

        # ---------------------------
        # Bottom panel: 2D hist
        # ---------------------------
        h = axes[1, i].hist2d(vy_log, T_log, bins=[vy_bins, T_bins],
                              weights=rho_log * vol / mass_cloud,
                              cmap=cm, norm=LogNorm(vmin=1e1, vmax=1e5), rasterized=True)

        axes[1, i].set_xscale('log')
        axes[1, i].set_yscale('log')
        axes[1, i].set_xlim(1e-5, 10)
        axes[1, i].set_ylim(9e3, 1e7)
        axes[1, i].set_xlabel(r'$v_{\mathrm{gas}} / v_w$')

        

        # ---------------------------
        # Top panel: 1D curve m_clump
        # ---------------------------
        # Compute m_clump (total mass per velocity bin)
        m_clump, edges = np.histogram(vy_log, bins=vy_bins, 
                                      weights=rho_log * vol / mass_cloud)
        
        
        # ---------------------------
        # Top panel: velocity histograms for phases
        # ---------------------------

        # Cold: T < 1e5 K
        m_cold, edges = np.histogram(
            vy_log[T_log < 1e5], bins=vy_bins,
            weights=rho_log[T_log < 1e5] * vol / mass_cloud
        )
        if i ==1:
            # Fit to the curve for the middle plot
            centers = np.sqrt(edges[:-1] * edges[1:])
            popt, _ = curve_fit(y, centers, m_cold, p0=[1e4, 1.0, 0.1, 2], maxfev=10000)
            x_fit = np.logspace(-5, 1, 200)
            y_fit = y(x_fit, *popt)
            axes[0, i].plot(x_fit, y_fit, 'k--')
            print(f"Fit parameters for plot {i}: A={popt[0]:.2e}, alpha={popt[1]:.2f}, xc={popt[2]:.2f}, beta={popt[3]:.2f}")

        # Warm: 1e5 <= T < 6e5 K
        m_warm, _ = np.histogram(
            vy_log[(T_log >= 1e5) & (T_log < 6e5)], bins=vy_bins,
            weights=rho_log[(T_log >= 1e5) & (T_log < 6e5)] * vol / mass_cloud
        )

        # Hot: T >= 6e5 K
        m_hot, _ = np.histogram(
            vy_log[T_log >= 6e5], bins=vy_bins,
            weights=rho_log[T_log >= 6e5] * vol / mass_cloud
        )

        # Geometric centers for log-spaced bins
        centers = np.sqrt(edges[:-1] * edges[1:])

        # Plot all three on the same axis
        axes[0, i].plot(centers, m_cold, color="blue", lw=1.5)
        axes[0, i].plot(centers, m_warm, color="green", lw=1.5)
        axes[0, i].plot(centers, m_hot,  color="red", lw=1.5)


        axes[0, i].plot(centers, m_clump, color="k", lw=1.5)
        axes[0, i].set_xscale("log")
        axes[0, i].set_yscale("log")
        axes[0, 0].set_ylabel(r"$m/m_{\mathrm{cl}}$")
        axes[0, i].set_xlim(1e-5, 10)
        axes[0, i].set_ylim(1e2, 1e6)
        # Hide x tick labels for top panel
        axes[0, i].tick_params(labelbottom=False)

    # Shared y-label for bottom row
    axes[1, 0].set_ylabel('T [K]')
    axes[0, 0].set_title(r'$t \sim  0.1\,t_\mathrm{ent}$', size=18, pad=10)
    axes[0, 1].set_title(r'$t \sim  0.5\,t_\mathrm{ent}$', size=18, pad=10)
    axes[0, 2].set_title(r'$t \sim  \,t_\mathrm{ent}$', size=18, pad=10)

    # Add colorbar to the right of all subplots
    cbar_ax = fig.add_axes([0.92, 0.20, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(h[3], cax=cbar_ax, label=r'$m_\mathrm{tot} / m_\mathrm{cl}$')

    fig.tight_layout(rect=[0, 0, 0.92, 1])  # leave space on right for colorbar
    save_path = f'/u/ferhi/Figures/T_v_plots/T_v_phase_marginals.pdf'
    #save_path = 'marginals.png'
    print("File saved in : ", save_path)
    plt.savefig(save_path, bbox_inches = 'tight', dpi=300)
    plt.close(fig)

if __name__ == '__main__':

    RUNS = [os.getcwd()]
    run_paths = ["/viper/ptmp/ferhi/LEGACY/fvLism/01kc/fv01_30r/"]
    parts = run_paths[0].split('/')
    saveFile = run_paths[0].split('ferhi/')[-1]

    if not os.path.exists(os.path.join('/u/ferhi/Figures/T_v_plots/',saveFile)): 
        os.makedirs(os.path.join('/u/ferhi/Figures/T_v_plots/',saveFile))
    print(saveFile)

    sim = SingleCloudCC(os.path.join(run_paths[0], 'ism.in'), dir=run_paths[0])
    code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
    code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))

    output_delay = int(float(sim.reader.get('parthenon/output0', 'dt'))/ float(sim.reader.get('parthenon/output1', 'dt')))
    v_boost = np.loadtxt(os.path.join(run_paths[0], 'out/parthenon.out1.hst'))[:, -1] 

    files = np.sort(glob.glob(os.path.join(run_paths[0], 'out/parthenon.prim.*.phdf')))[1:-1]
    indexes = output_delay * np.asarray(range(1,len(files)))

    v_correction = v_boost[indexes]
    
    rho_cloud_cgs = float(sim.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
    m_cloud = 4 * np.pi / 3 * (0.1*sim.R_cloud)**3 * rho_cloud_cgs
    
    run_list_multiplot = [
        "/viper/ptmp/ferhi/LEGACY/fvLism/01kc/fv01_30r/out/parthenon.prim.00001.phdf", 
        "/viper/ptmp/ferhi/LEGACY/fvLism/01kc/fv01_30r/out/parthenon.prim.00050.phdf",
        "/viper/ptmp/ferhi/LEGACY/fvLism/01kc/fv01_30r/out/parthenon.prim.00100.phdf", 
    ]
    
    v_correction_list_multiplot =  np.asarray([1,18,100])
    
    T_v_phase_multi(run_list_multiplot, vol =code_length_cgs**3, mass_cloud = m_cloud, v_correction_list = v_correction_list_multiplot, v_wind = sim.v_wind, output_dir = saveFile )

    #for j, file in enumerate(files):
    #    print('Processing file:', file)
    #    v_correction_single = v_correction[j] * code_length_cgs / code_time_cgs
    #    T_v_phase(file, vol = code_length_cgs**3, v_correction=v_correction_single, v_wind = sim.v_wind,output_dir = saveFile)

