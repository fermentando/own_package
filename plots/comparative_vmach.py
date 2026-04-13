import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stratified_box import StratifiedBox
from matplotlib.colors import LogNorm

# --- your existing functions (unchanged) ---
def read_hst(run): 
    data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
    data = np.where(data == 0, 1e-22, data)
    return data

def vel_evolution(run):
    data = read_hst(run)
    vel_ind = 13
    mass_ind = 10
    velocity = -(data[:, vel_ind] / data[:, mass_ind])
    return velocity


# --- helper: extract parameters ---
def get_sim_params(run):

    sim = StratifiedBox(os.path.join(run, 'strat.in'), dir=run)

    try:
        mach = float(sim.reader.get('problem/turbulence', 'Mach_drive'))
    except:
        mach = np.nan

    # --- try to get radius ---
    radius = np.nan

    # Option 1: from config
    try:
        radius = float(run.split('/')[-3].split('r')[-1])
        
    except:
        radius = float(sim.reader.get('problem/stratified_box', 'r_cloud_inserted'))

    # Option 2: parse from folder name (e.g. R10, r5, etc.)
    if np.isnan(radius):
        name = os.path.basename(os.path.normpath(run))
        import re
        match = re.search(r'[Rr](\d+)', name)
        if match:
            radius = float(match.group(1))

    return mach, radius


# --- main ---
def main(base_dir):

    runs = glob.glob(os.path.join(base_dir, '*', '*/'))

    machs = []
    velocities = []
    radii = []
    vel_errors = []

    for run in runs:
        if 'r0.1' in run: continue
        if 'old' in run: continue
        try:
            vel = vel_evolution(run)
            mach, radius = get_sim_params(run)

            if len(vel) == 0 or np.isnan(mach):
                continue

            avg_vel = np.max(vel)
            vel_err = np.std(vel) 

            machs.append(mach)
            velocities.append(avg_vel)
            radii.append(radius)
            

            print(f"{run} | Mach={mach:.2f}, R={radius}, <v>={avg_vel:.3e}")

        except Exception as e:
            print(f"Skipping {run}: {e}")
            continue

    machs = np.array(machs)
    velocities = np.array(velocities)
    radii = np.array(radii)
    vel_errors.append(vel_err)

    print("these are the machs: ", machs)
    print("these are the velocities: ", velocities)

    # --- plotting ---
    plt.figure(figsize=(8,6))

    sc = plt.scatter(
        machs,
        velocities/ (machs * 150),
        c=radii,
        cmap='viridis',
        norm=LogNorm(vmin=10, vmax=1e4),  # <-- log normalization here
        s=60,
        edgecolor='k'
    )

    #plt.errorbar(
    #    machs,
    #    velocities,
    #    yerr=vel_errors,   # <-- array of errors you computed
    #    fmt='none',
    #    ecolor='gray',
    #    alpha=0.6,
    #    capsize=3,
    #    zorder=2
    #)


    cbar = plt.colorbar(sc)
    cbar.set_label("Radius (pc)")

    plt.xlabel("Mach number")
    plt.ylabel(r"$v_{peak}/v_{turb}$")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    saveDir = '/u/ferhi/Figures/comparative_vmach_average.png'
    print(f'Saving figure to {saveDir}')
    plt.savefig(saveDir)


if __name__ == "__main__":
    base_dir = "/viper/ptmp/ferhi/InfallTurbulent/mach_test"  # <-- change this
    main(base_dir)