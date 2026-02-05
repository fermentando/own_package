import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from stratified_box import StratifiedBox
from turbulent_box import TurbulentBox
from cooling import get_c_s


# -------------------------------
# Utilities
# -------------------------------
def read_hst(run):
    data = np.loadtxt(os.path.join(run, "out/parthenon.out1.hst"))
    data = np.where(data == 0, 1e-22, data)
    return data


def vel_evolution(run):
    data = read_hst(run)
    t = data[:, 0]
    vel = data[:, 13] / data[:, 10]  # momentum / mass
    return t, vel


def extract_r_value(path):
    m = re.search(r"r([\d.]+)", path)
    return float(m.group(1)) if m else None


def get_mach_at_time(run_dir, target_teddy=1.2):
    data = read_hst(run_dir)

    try:
        run = StratifiedBox(os.path.join(run_dir, "strat.in"), run_dir)
        T0 = float(run.reader.get("problem/stratified_box", "T_base"))
    except:
        run = TurbulentBox(os.path.join(run_dir, "turbulence.in"), run_dir)
        rho0 = float(run.reader.get("problem/turbulence", "rho0")) * run.code_mass_cgs / run.code_length_cgs**3
        p0 = float(run.reader.get("problem/turbulence", "p0")) * run.code_mass_cgs / run.code_length_cgs / run.code_time_cgs**2
        T0 = p0 / (rho0 * run.kb / run.mbar)

    mach_drive = float(run.reader.get("problem/turbulence", "Mach_drive"))

    Lxmin = float(run.reader.get("parthenon/mesh", "x1min"))
    Lxmax = float(run.reader.get("parthenon/mesh", "x1max"))
    L_drive = Lxmax - Lxmin

    cs = get_c_s(T0)
    v_turb = cs * mach_drive / (run.code_length_cgs / run.code_time_cgs)
    t_eddy = L_drive / v_turb

    t_norm = data[:, 0] / t_eddy
    idx = np.argmin(np.abs(t_norm - target_teddy))

    mach = data[idx, -1] / (
        (Lxmax - Lxmin)
        * (float(run.reader.get("parthenon/mesh", "x2max")) - float(run.reader.get("parthenon/mesh", "x2min")))
        * (float(run.reader.get("parthenon/mesh", "x3max")) - float(run.reader.get("parthenon/mesh", "x3min")))
    )

    return mach, t_eddy


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    run_paths = sorted(glob.glob("/viper/ptmp/ferhi/StratDisk/m0.*/r*/burnin"))
    print(run_paths)

    mach_list = []
    z_list = []
    zerr_list = []
    r_list = []

    for run in run_paths:

        r_cloud = extract_r_value(run)
        if r_cloud is None:
            continue

        try:
            mach, t_eddy = get_mach_at_time(run, target_teddy=1.2)
            sim = StratifiedBox(os.path.join(run, "strat.in"), dir=run)
            code_time = sim.code_time_cgs
            code_length = sim.code_length_cgs

            t, v = vel_evolution(run)

            # convert to physical units
            t_myr = t * code_time / u.Myr.to("s")
            v_kms = v * code_length / code_time / 1e5

            # remove initial velocity artifact
            mask = np.abs(v_kms - 1.0) > 0.1
            t_myr = t_myr[mask]
            v_kms = v_kms[mask]

            t_myr -= t_myr[0]

            # evaluate up to target time
            t_target = 1.2 * t_eddy * code_time / u.Myr.to("s")
            sel = t_myr <= t_target

            t_sel = t_myr[sel]
            v_sel = v_kms[sel]

            # integrate position
            t_s = t_sel * u.Myr.to("s")
            v_cms = v_sel * 1e5

            z = np.trapz(v_cms, t_s) / u.pc.to("cm")
            z_norm = z / r_cloud

            # velocity dispersion → position error
            sigma_v = np.std(v_cms)
            dt = np.diff(t_s)
            sigma_z = np.sqrt(np.sum((sigma_v * dt) ** 2)) / u.pc.to("cm")
            sigma_z_norm = sigma_z / r_cloud

            mach_list.append(mach)
            z_list.append(z_norm)
            zerr_list.append(sigma_z_norm)
            r_list.append(r_cloud)

            print(f"{run}: Mach={mach:.3f}, z/r={z_norm:.2f} ± {sigma_z_norm:.2f}")
        except Exception as e:
            print(f"Error extracting evolution for run {run}: {e}.")
            

    # -------------------------------
    # Plot
    # -------------------------------

    from matplotlib.colors import LogNorm

    r_array = np.array(r_list)
    mach_array = np.array(mach_list)
    z_array = np.array(z_list)
    zerr_array = np.array(zerr_list)

    norm = LogNorm(vmin=0.1, vmax=100)
    cmap = plt.cm.viridis


    sc = plt.scatter(
        mach_array,
        z_array,
        c=r_array,
        cmap=cmap,
        norm=norm,
        s=70,
        edgecolor="k",
        zorder=3
    )

    plt.errorbar(
        mach_array,
        z_array,
        yerr=zerr_array,
        fmt="none",
        ecolor="k",
        alpha=0.6,
        zorder=2
    )

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$r_{\rm cloud}$ [pc]")

    plt.xlabel(r"Mach number")
    plt.ylabel(r"$z / r_{\rm cloud}$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    print("Saving file to position_vs_mach.png")
    plt.savefig("position_vs_mach.png", dpi=300)
    plt.show()
