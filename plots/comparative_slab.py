import numpy as np
import matplotlib.pyplot as plt
from adjust_ics import SingleCloudCC
import glob
import os

run_paths = [
    '/viper/ptmp2/ferhi/fvLism/01kc/fv01_30r',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv02',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv02_destr',
    '/viper/ptmp2/ferhi/fvLism/kc/fv01_shorter',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv03_long',
    '/viper/ptmp2/ferhi/fvLism/02kc/fv03',
    '/viper/ptmp2/ferhi/fvLism/01kc/fv01_scaleless'
]

plt.figure(figsize=(8, 5))

for j, run in enumerate(run_paths):
    run_name = run  
    print(run)
            
    sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
    code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
    files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
    parts = run.split('/')
    saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
    results_file = os.path.join('/u/ferhi/Figures/', saveFile, "cold_box_y_extent.npz")

    # If file exists, load it
    if os.path.exists(results_file):
        existing = np.load(results_file, allow_pickle=True)
        snapshot_indices = list(existing['snapshot_indices'])
        filenames = list(existing['filenames'])
        y_extents = list(existing['y_extents'])
        start_idx = len(snapshot_indices)
        print(f"Resuming from snapshot {start_idx}")
    
    kc = run.split('kc')[0][-2:]
    if kc == '01':
        r_over_rcrit = 1
    elif kc == '02':
        r_over_rcrit = 10
    else:
        r_over_rcrit = 0.1   
    cloud_p = 0.1 * sim.Rcrit_x_surv_ratio * (sim.tcoolmix/sim.tcc)
    x1max, x1min, nx1 = float(sim.reader.get('parthenon/mesh', 'x1max')), float(sim.reader.get('parthenon/mesh', 'x1min')), float(sim.reader.get('parthenon/mesh', 'nx1'))
    dt = float(sim.reader.get('parthenon/output1', 'dt'))
    fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
    depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
    
    
    t_scale =  depth * sim.R_cloud * fv / sim.v_wind * dt
    d_scale = (x1max - x1min) /  nx1 * float(sim.reader.get('units', 'code_length_cgs'))
    data = np.load(results_file)

    
    plt.plot(data['snapshot_indices'] * t_scale, data['y_extents'] / data['y_extents'][0],  label=run_name)
plt.yscale('log')
plt.xscale('linear')
plt.xlabel("Snapshot Index (time)")
plt.ylabel(r'$L [ r_{cl} ]$')
plt.grid(True)
plt.legend()
#plt.ylim(1e-1, 1e2)
plt.tight_layout()

fig_path = os.path.join('/u/ferhi/Figures/cold_box_y_evolution.png')
plt.savefig(fig_path)
print("Saved plot to:", fig_path)