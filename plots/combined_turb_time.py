import os
import numpy as np
import matplotlib.pyplot as plt
from adjust_ics import *
import utils as ut
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import gridspec
import seaborn as sns
import matplotlib.colors as mcolors

cmap = sns.color_palette("viridis", as_cmap=True)  # or "magma", "plasma", etc.
norm = mcolors.LogNorm(vmin=1e-2, vmax=10)  # Log scale if range is wide
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
linestyles = {1: '-', 2: ':', 3: '-.', 0: '--'}

def hdf_turb(run, mode, cache_file='fix_vturb.npz'):
    path = os.path.join(run, f'{mode}_' + cache_file)
    if os.path.exists(path):
        print(f"Loading cached data from {path}")
        data = np.load(path)
        return data['t'], data['vturb'], data['vturb_err_lower'], data['vturb_err_upper']

    #else:
        #raise FileNotFoundError(f"Cache file not found for {run} in mode '{mode}'")
def plot_vturb_all_runs(run_list, outdir):
    import matplotlib.pyplot as plt
    plt.style.use("custom_plot")

    # Create figure and GridSpec
    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    # Plot axes (top row)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]



    modes = ['cold', 'hot']

    for m, mode in enumerate(modes):
        for run in run_list:
            sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
            dt = float(sim.reader.get('parthenon/output0', 'dt'))
            depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
            fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
            tccfact = 0.1 * sim.tcoolmix/sim.tcc * depth if sim.tcoolmix / sim.tcc >= 0.1 else 0.1

            cs_gas = np.sqrt(5 / 3 * ut.constants.kb * sim.T_cloud / sim.mbar)

            base_fv = int(run.split('fv')[-1][:2])
            fv = 10 ** (-base_fv)
            linestyle = linestyles.get(base_fv, '-')
            color = sm.to_rgba(tccfact)

            try:
                result = hdf_turb(run, mode)
                if result is None:
                    print(f"[WARNING] hdf_turb returned None for run: {run}")
                    continue 
                t, vturb, vturb_err_lower, vturb_err_upper = result
                tsh = t * dt

                ax[m].plot(tsh, vturb / cs_gas, color=color, linestyle=linestyle, alpha=0.9)
                ax[m].fill_between(
                    tsh,
                    (vturb - vturb_err_lower) / cs_gas,
                    (vturb + vturb_err_upper) / cs_gas,
                    color=color, alpha=0.15
                )
            except FileNotFoundError as e:
                print(e)

        ax[m].set_xlim(right=20)
        ax[m].grid(True)
        ax[m].set_xlabel(r'$t [\tilde{t}_\mathrm{cc}]$')
        ax[m].set_ylim(top=5)

    ax[0].set_ylabel(r'$v_{\mathrm{turb}} / c_{\mathrm{s,cold}}$')

    # Add colorbar
    sm.set_array([]) 
    cax = fig.add_axes([0.15, 0.95, 0.7, 0.03])  # [left, bottom, width, height] in figure fraction
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(r'$L_{\mathrm{ISM}}$ [$r_{\mathrm{cl}}$]', labelpad=10)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    #cbar.ax.tick_params(colors='white', which='both', labelcolor='black')
    plt.setp(ax[1].get_yticklabels(), visible=False)

    # Add linestyle legend for fv
    fv_legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label=r'$f_v = 10^{-1}$'),
        Line2D([0], [0], color='black', linestyle=':', label=r'$f_v = 10^{-2}$'),
    ]
    fig.subplots_adjust(top=0.90)
    ax[1].legend(
        handles=fv_legend_elements,
        loc='upper right',
        #ncol=2,
        #bbox_to_anchor=(0.5, 1.05),
        frameon=True
    )

    # Save
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "yav_vturb_cold_hot.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    runs = [
        '/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/',
        '/viper/ptmp/ferhi/fvLism/01kc/fv02/',
        '/viper/ptmp/ferhi/fvLism/01kc/fv02_destr/',
        '/viper/ptmp/ferhi/fvLism/kc/fv01_shorter/',  
    ]
    
    plot_vturb_all_runs(runs, outdir='/u/ferhi/Figures/')