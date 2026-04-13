"""
clump_analysis.py
-----------------
Reads an AthenaPK HDF5 snapshot, identifies cold gas clumps below a temperature
threshold, computes average vel_2 per clump, and plots vel2 vs clump size.

Usage:
    python clump_analysis.py --file path/to/snapshot.athdf [--T_thresh 1e4] [--output plot.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum as ndi_sum

from read_hdf5 import read_hdf5


# =============================================================================
# Parameters
# =============================================================================

DEFAULT_T_THRESH = 1e5   # K — cold gas threshold
DEFAULT_FILE     = "/viper/ptmp/ferhi/StratDisk/noturb/r100/out/parthenon.prim.00009.phdf"
DEFAULT_OUTPUT   = "vel2_vs_clumpsize.png"


# =============================================================================
# Clump finding
# =============================================================================

def find_clumps(temperature, T_thresh):
    """
    Label connected cold regions where T < T_thresh.

    Returns
    -------
    labeled : ndarray (int)
        Array of clump labels (0 = not a clump).
    n_clumps : int
        Number of distinct clumps found.
    """
    cold_mask = temperature < T_thresh
    struct = np.ones((3,) * temperature.ndim, dtype=int)  # full connectivity
    labeled, n_clumps = label(cold_mask, structure=struct)
    return labeled, n_clumps


def compute_clump_stats(labeled, n_clumps, vel2, density):
    """
    For each clump compute:
      - size      : number of cells  (directly from scipy label indexing)
      - vel2_mean : density-weighted mean of vel_2
      - vel2_std  : density-weighted std of vel_2 (used as errorbar)

    Returns
    -------
    sizes, vel2_means, vel2_stds : 1-D arrays of length n_clumps
    """
    clump_ids = np.arange(1, n_clumps + 1)

    # Cell counts per clump — scipy gives this cheaply via ndi_sum on ones
    sizes = (ndi_sum(np.ones_like(labeled), labeled, clump_ids).astype(int))**(1/3.)

    # Density-weighted vel2 stats
    rho_sum    = ndi_sum(density,        labeled, clump_ids)
    vel2_means = ndi_sum(density * vel2, labeled, clump_ids) / rho_sum

    # Weighted std: sqrt( sum(w*(v - mean)^2) )
    vel2_stds = np.sqrt(
        ndi_sum(density * (vel2 - vel2_means[labeled - 1]) ** 2, labeled, clump_ids)
        / rho_sum
    )

    return sizes, vel2_means, vel2_stds


# =============================================================================
# Plotting
# =============================================================================

def plot_vel2_vs_size(sizes, vel2_means, vel2_stds, T_thresh, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        sizes, vel2_means/1e5, yerr=vel2_stds/1e5,
        fmt='o', markersize=4, capsize=3, elinewidth=0.8,
        color='steelblue', alpha=0.8,
        label=rf"$T < {T_thresh:.0e}$ K"
    )

    ax.plot(np.linspace(sizes.min(), sizes.max(), 100), 10 * np.linspace(sizes.min(), sizes.max(), 100)**(3/4), 'k--', label=r"$\propto t_grow")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Clump size  [cells]", fontsize=13)
    ax.set_ylabel(r"$\langle v_2 \rangle$  [km/s]", fontsize=13)
    ax.set_title(r"Clump $v_2$ vs clump size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[clump_analysis] Plot saved → {output_path}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AthenaPK cold-clump vel2 analyser")
    parser.add_argument("--file",     default=DEFAULT_FILE,     help="Path to .athdf snapshot")
    parser.add_argument("--T_thresh", default=DEFAULT_T_THRESH, type=float,
                        help="Temperature threshold for cold gas (default: 1e4)")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT,   help="Output plot filename")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load fields
    # ------------------------------------------------------------------
    print(f"[clump_analysis] Reading {args.file} …")
    data = read_hdf5(args.file, fields=['T', 'rho', 'vel2'])

    # Mid-plane slice (same convention as your existing code)
    density     = data['rho']
    temperature = data['T'] 
    vel2        = data['vel2']

    print(f"  Array shape after slicing: {density.shape}")

    # ------------------------------------------------------------------
    # 2. Find cold clumps
    # ------------------------------------------------------------------
    print(f"[clump_analysis] Finding clumps with T < {args.T_thresh:.2e} …")
    labeled, n_clumps = find_clumps(temperature, args.T_thresh)
    print(f"  Found {n_clumps} clumps")

    if n_clumps == 0:
        print("[clump_analysis] No clumps found — try raising T_thresh.")
        return

    # ------------------------------------------------------------------
    # 3. Compute per-clump statistics
    # ------------------------------------------------------------------
    sizes, vel2_means, vel2_stds = compute_clump_stats(labeled, n_clumps, vel2, density)
    print(f"  Clump sizes : min={sizes.min()}  max={sizes.max()}  "
          f"median={int(np.median(sizes))} cells")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    plot_vel2_vs_size(sizes, vel2_means, vel2_stds, args.T_thresh, args.output)


if __name__ == "__main__":
    main()