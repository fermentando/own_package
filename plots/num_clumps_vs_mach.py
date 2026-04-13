from cProfile import label

import yt
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from stratified_box import StratifiedBox
from scipy.ndimage import label
from yt.data_objects.level_sets.api import Clump, find_clumps
import matplotlib.cm as cm
from read_hdf5 import read_hdf5

base_run_paths = [
    "/viper/ptmp/ferhi/InfallTurbulent/mach_test/m0.1/500lshatter/",
    "/viper/ptmp/ferhi/InfallTurbulent/mach_test/m0.3/500lshatter/",
    "/viper/ptmp/ferhi/InfallTurbulent/mach_test/m0.5/500lshatter/",
]

times = [0, 1,2,3]
cmap = cm.get_cmap('viridis')

for path in base_run_paths:
    print(f"Processing {path}...")
    sim = StratifiedBox(path + "strat.in", path)
    dt = float(sim.reader.get('parthenon/output2', 'dt'))

    for i, time in enumerate(times):
        shift_time = time * sim.t_cc / sim.code_time_cgs
        snp_index = int(shift_time / dt)
        print("This is the snapshot index:", snp_index)
        snp_path = os.path.dirname(path) + f"/out/parthenon.prim.{40 + snp_index:05d}.phdf"
        ds = read_hdf5(snp_path, ['T'])

        mask = ds['T'] < 1e5
        labeled_array, num_clumps = label(mask)
        color = cmap(i / len(times))
        plt.scatter(sim.mach, num_clumps, color=color, label=f't={time}')

plt.yscale('log')
#plt.xscale('log')
plt.ylim(bottom=1)

plt.xlabel('Mach number')
plt.ylabel('Number of clumps')



saveDir = '/u/ferhi/Figures/clumps_vs_mach.png'
print(f"Figure saved to {saveDir}")
plt.savefig(saveDir)