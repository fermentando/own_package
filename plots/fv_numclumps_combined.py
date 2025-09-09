import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from adjust_ics import SingleCloudCC
from utils import get_user_args, get_n_procs_and_user_args
from read_hdf5 import read_hdf5
from mass_evolution import hst_evolution
# --- Config ---
run_paths = [
    '/viper/ptmp2/ferhi/fvLism/kc/fv01_shorter',
    '/viper/ptmp2/ferhi/fvLism/02kc/fv01',
    #'/viper/ptmp2/ferhi/fvLism/10kc/fv01_v2',
    #'/viper/ptmp2/ferhi/fvLism/01kc/fv01_30r',

]

fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(4, 1, height_ratios=[3, 3,3,3], hspace=0.25)
axes_main = [fig.add_subplot(gs[i, 0]) for i in range(4)]
colors = ['#008080',	'#FF6F61']

for j, run in enumerate(run_paths):
    sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)

    fvfa_file = os.path.join('/u/ferhi/Figures/', *run.split('/')[-3:], "Analysis/fv_fA.npz")
    clump_file = os.path.join('/u/ferhi/Figures/clump_distribution/', *run.split('/')[-3:], "Analysis/num_clumps_over_time.npz")
    rclump_file = os.path.join('/u/ferhi/Figures/clump_distribution/', *run.split('/')[-3:], "Analysis/rcrit_num_clumps_over_time.npz")
    hst_file = os.path.join('/viper/ptmp2/ferhi/', *run.split('/')[-3:])
    
    if not os.path.exists(fvfa_file):
        fvfa_file = os.path.join('/u/ferhi/Figures/', *run.split('/')[-3:], "fv_fA.npz")

    if os.path.exists(fvfa_file):
        data = np.load(fvfa_file)
        code_units_time = float(sim.reader.get('units', 'code_time_cgs'))
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
        tsh = depth * sim.R_cloud / sim.v_wind

        # Load snapshot times
        times = []
        for idx in data['snapshot_indices']:
            try:
                with h5py.File(os.path.join(run, 'out', f'parthenon.prim.{str(idx).zfill(5)}.phdf'), 'r') as f:
                    times.append(f['Info'].attrs['Time'] * code_units_time)
            except FileNotFoundError:
                with h5py.File(os.path.join(run, 'out', 'parthenon.prim.final.phdf'), 'r') as f:
                    times.append(f['Info'].attrs['Time'] * code_units_time)

        t = np.array(times) / tsh
        fv_vals = data['fv_values']

        axes_main[0].plot(t, fv_vals, color=colors[j])

    if os.path.exists(clump_file):
        clump_data = np.load(clump_file)
        clump_times = clump_data['times'] / tsh
        num_clumps = clump_data['num_clumps']
        axes_main[1].plot(clump_times, num_clumps, color=colors[j])
        axes_main[1].set_yscale('log')
        axes_main[1].set_ylim(1e2, 5e3)

    if os.path.exists(rclump_file):
        clump_data = np.load(rclump_file)
        clump_times = clump_data['times'] / tsh
        num_clumps = clump_data['num_clumps']/clump_data['num_clumps'][-1]  # Normalize to first snapshot
        axes_main[2].plot(clump_times, num_clumps, color=colors[j])

    if os.path.exists(hst_file):
        times, mc, _, _, _ = hst_evolution(hst_file)
        times = times * code_units_time / tsh
        mdot = np.gradient(mc, times) 

        axes_main[3].plot(times, mdot, color=colors[j])

# === Label each subplot ===
axes_main[0].set_ylabel(r'$f_\mathrm{V}$ ')
axes_main[1].set_ylabel(r'$N_\mathrm{c}$')
axes_main[2].set_ylabel(r'$N_\mathrm{c} (r > r_\mathrm{crit}) / N_\mathrm{final}$')
axes_main[3].set_ylabel(r'$\dot{m} \, [m_\mathrm{c}/t_{\mathrm{sh}}]$')
axes_main[3].set_xlabel(r'$t/t_{\mathrm{sh}}$')

plt.tight_layout()
plt.savefig("/u/ferhi/Figures/combined_fv_clumps_plot.png", dpi=300)
print("Plot saved to /u/ferhi/Figures/combined_fv_clumps_plot.png")
plt.show()
