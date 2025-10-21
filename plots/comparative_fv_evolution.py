import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm
from adjust_ics import SingleCloudCC
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
import colorsys
from matplotlib.ticker import LogLocator, SymmetricalLogLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator

plt.style.use('custom_plot')
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{cancel}"
})

# --- Config ---
run_paths = [
    '/viper/ptmp2/ferhi/LEGACY/fvLism/01kc/fv02',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/kc/fv01_shorter',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/01kc/fv03_long',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/02kc/fv03',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/02kc/fv02',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/02kc/fv01',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/10kc/fv01_v2',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/01kc/fv01_30r',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/kc/fv01',
    '/viper/ptmp2/ferhi/LEGACY/fvLism/01kc/fv01_scaleless',
    '/viper/ptmp/ferhi/LEGACY/fvLism/10kc/fv01_v2'
]



linestyles = {1: '-', 3: '--', 0: '-.', 2: ':'}

kc_map = {}
for run in run_paths:
    sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
    depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
    kc_str = run.split('kc')[0].split('/')[-1]
    kc_val = 10 ** (1 - float(kc_str)) if kc_str else 10
    kc_map[run] = int(round(-np.log10(kc_val)))

def boost_saturation(color, sat_mult=1.3, val_mult=1.1):
    """Increase saturation and brightness in HSV space."""
    r, g, b = to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(s * sat_mult, 1.0)
    v = min(v * val_mult, 1.0)
    return to_hex(colorsys.hsv_to_rgb(h, s, v))

# Get seaborn colors
blue_raw = sns.color_palette("Blues", 8)[-1]
orange_raw = sns.color_palette("Oranges", 8)[-1]

# Boosted extremes
blue = boost_saturation(blue_raw, sat_mult=1.4, val_mult=1.1)
orange = boost_saturation(orange_raw, sat_mult=1.4, val_mult=1.1)

# Strong beige center
beige = "#e6cfa5"

# Build colormap
colors = [blue, beige, orange]
cmap = LinearSegmentedColormap.from_list("blue_beige_orange", colors, N=256)

#cmap = cm.get_cmap('vanimo')
norm = LogNorm(vmin=10, vmax=1e4)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# --- Figure setup ---
fig = plt.figure(figsize=(8, 8))  # narrower width since one column only
gs = gridspec.GridSpec(6, 3, width_ratios=[1, 0.04, 1], height_ratios=[0.18, 0.4, 0.01, 1, 2, 2], hspace=0.14)

ax0 = fig.add_subplot(gs[3:, 0])
ax1 = fig.add_subplot(gs[3, 2])
ax2 = fig.add_subplot(gs[4:, 2])
axes_main = [ax1, ax0, ax2]

# --- Plot each run ---
for run in run_paths:
    sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
    depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
    kc_key = kc_map[run]
    #color = cmap(norm(10 ** -kc_key))
    color = cmap(norm(10 * depth))

    base_path = os.path.join('/u/ferhi/Figures/', *run.split('/')[-3:])
    results_file = os.path.join(base_path, "Analysis/fv_fA.npz")
    if not os.path.exists(results_file):
        results_file = os.path.join(base_path, "fv_fA.npz")

    if os.path.exists(results_file):
        data = np.load(results_file)
        dt = float(sim.reader.get('parthenon/output1', 'dt'))
        dt2 = float(sim.reader.get('parthenon/output0', 'dt'))
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
        code_units_time = float(sim.reader.get('units', 'code_time_cgs'))
        dt = float(sim.reader.get('parthenon/output1', 'dt'))
        dt2 = float(sim.reader.get('parthenon/output0', 'dt'))
        
        
        
        tsh =  depth * sim.R_cloud / sim.v_wind

        t1 = sim.R_cloud / sim.v_wind
        t2 = 10 * fv * depth *  sim.R_cloud / sim.v_wind

        # Linear sum
        t_linear = t1 + t2

        times_f = []
        dt_hdf = float(sim.reader.get('parthenon/output0', 'dt'))
        snn= np.array(list(range(len(data['snapshot_indices']))))
        print(len(snn))
        times_fa = dt_hdf * snn / depth *10
        times_fv = dt_hdf * snn / depth * 10 #(0.1 + fv * depth)


        fa = data['fa_values']
        fv_vals = data['fv_values']

        linestyle = linestyles.get(int(-np.log10(fv)), '-')
        try:
            if "scaleless" in run: 
                axes_main[0].plot(times_fa, fa, color="black", label = r'$\cancel{r_\mathrm{cl}} $')
                axes_main[1].plot(times_fv, fv_vals, color="black", label = r'$\cancel{r_\mathrm{cl}} $')
            else:
                axes_main[0].plot(times_fa, fa, color=color, linestyle=linestyle)
                axes_main[1].plot(times_fv, fv_vals, color=color, linestyle=linestyle)
        except Exception as e:
            print(f"Error plotting {run}: {e}")
            continue
            

    cold_file = os.path.join(base_path, "cold_box_y_extent.npz")
    if os.path.exists(cold_file):
        box_data = np.load(cold_file)

        times_y = []
        dt_hdf = float(sim.reader.get('parthenon/output0', 'dt'))
        snn= np.array(list(range(len(box_data['snapshot_indices']))))
        print(len(snn))
        times_y =  dt_hdf * snn / depth * 10
        y_norm = box_data['y_extents'] / 8  / 100
        linestyle = linestyles.get(int(-np.log10(fv)), '-')
        if "scaleless" in run: axes_main[2].plot(times_y, y_norm, color="black", label = r'$\cancel{r_\mathrm{cl}} $')
        else: axes_main[2].plot(times_y, y_norm, color=color, linestyle=linestyle)

# --- Formatting ---
axes_main[1].set_yscale('log')
axes_main[2].set_yscale('log')
axes_main[1].legend(frameon=True, loc='upper right', fontsize=16)


axes_main[2].set_xlabel(r'$t_\mathrm{sh}$')
axes_main[1].set_xlabel(r'$t_\mathrm{sh}$')
axes_main[0].set_ylabel(r'$f_A$', labelpad=8)
axes_main[1].set_ylabel(r'$f_v$', labelpad=8)
axes_main[2].set_ylabel(r'$\ell_\mathrm{slab} / \chi r_\mathrm{cl,init}$', labelpad=-2)

axes_main[0].set_ylim(0.3, 1.1)
axes_main[1].set_ylim(1e-4, 1)
axes_main[2].set_ylim(0.1, 100)

# Remove x tick labels on top two panels
for ax in [axes_main[0]]:
    ax.tick_params(labelbottom=False)

#for ax in [axes_main[0], axes_main[2]]:
axes_main[2].set_xscale("symlog", linthresh=10)
axes_main[1].set_xscale("symlog", linthresh=10)
axes_main[0].set_xscale("symlog", linthresh=10)
for ax in [axes_main[1]]:
    ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10.0, linthresh=10, subs=None))
    ax.xaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=10, subs=np.arange(2,10)*0.1))
    ax.set_xlim(0, 50)
    # Add integer ticks from 10 to 15
    ticks = list(ax.get_xticks()) + list(range(10, 50))
    ax.xaxis.set_ticks(ticks)
    ax.xaxis.set_tick_params(which='major', labelsize=17, width=1, length=4)
    # Remove tick labels for ticks above 10
    labels = [str(int(t)) if t <= 10 else "" for t in ticks]
    ax.set_xticklabels(labels)
    ax.grid(False)

for ax in [axes_main[0], axes_main[2]]:
    ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10.0, linthresh=10, subs=None))
    ax.xaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=10, subs=np.arange(2,10)*0.1))
    ax.set_xlim(0, 40)
    # Add integer ticks from 10 to 15
    ticks = list(ax.get_xticks()) + list(range(10, 40))
    ax.xaxis.set_ticks(ticks)
    ax.xaxis.set_tick_params(which='major', labelsize=17, width=1, length=4)
    # Remove tick labels for ticks above 10
    labels = [str(int(t)) if t <= 10 else "" for t in ticks]
    ax.set_xticklabels(labels)
    ax.grid(False)


# Offset x label slightly
ax_cb = fig.add_subplot(gs[0, :])  # last row, all columns

# Add horizontal colorbar inside this axis
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
cbar.set_label(r'$L_\mathrm{ISM} [r_\mathrm{cl}]$', labelpad=2, y=1.1, size = 18)  # y>1 moves label above
cbar.ax.xaxis.set_label_position('top')  # move label to top
cbar.ax.xaxis.tick_top()   
cbar.ax.tick_params(which='both', color='white', labeltop=True, labelbottom=False)


# Legend for fv linestyles
from matplotlib.lines import Line2D
fv_legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label=r'$f_v = 10^{\mathrm{-1}}$'),
    Line2D([0], [0], color='black', linestyle=':', label=r'$f_v = 10^{\mathrm{-2}}$'),
    Line2D([0], [0], color='black', linestyle='--', label=r'$f_v = 10^{\mathrm{-3}}$'),
]
fig.subplots_adjust(top=0.93)  
fig.legend(
    handles=fv_legend_elements,
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, 0.82),  # Slightly above the plot
    frameon=True,
)

# --- Save and Show ---
plt.tight_layout()
fig_path = '/u/ferhi/Figures/modified_fv_fA_LISM.pdf'
plt.savefig(fig_path, bbox_inches='tight', dpi=300)
print("Saved plot to:", fig_path)
plt.show()
"""
run_paths = [
    '/viper/ptmp2/ferhi/fvLism/01kc/fv02',
    '/viper/ptmp2/ferhi/fvLism/kc/fv01_shorter',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv03_long',
    '/viper/ptmp2/ferhi/fvLism/02kc/fv03',
    '/viper/ptmp2/ferhi/fvLism/02kc/fv02',
    '/viper/ptmp2/ferhi/fvLism/02kc/fv01',
    '/viper/ptmp2/ferhi/fvLism/10kc/fv01_v2',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv01_30r',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv01_scaleless'
]"""