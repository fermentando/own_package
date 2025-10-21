import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy import stats
from joblib import Parallel, delayed
import argparse
import os
import seaborn as sns
import pickle

plt.style.use("custom_plot")
colours = sns.color_palette()

def compute_cluster_sizes(binary_field):
    labeled_array, num_features = label(binary_field)
    sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (label 0)
    return sizes


def simulate_percolation(dimensions, p, sigma):
    field = np.random.rand(*dimensions)
    smoothed_field = gaussian_filter(field, sigma=sigma)
    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    return smoothed_field > threshold


def get_cache_filename(p, sigma, grid_dims, cache_dir="results_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"fit_p{p:.2e}_sigma{sigma:.2f}_{grid_dims[0]}x{grid_dims[1]}x{grid_dims[2]}.pkl"
    return os.path.join(cache_dir, fname)

def lognorm_fit_simulation(p, sigma, grid_dims):
    cache_file = get_cache_filename(p, sigma, grid_dims)

    # Try to load cached result
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print(f"[Cache] Loaded result from {cache_file}")
            return pickle.load(f)

    # Otherwise, run simulation
    try:
        binary_field = simulate_percolation(grid_dims, p, sigma).astype(int)
        cluster_sizes = compute_cluster_sizes(binary_field)
        if len(cluster_sizes) == 0:
            raise ValueError("No clusters found.")

        r_clusters = (3 / (4 * np.pi) * cluster_sizes) ** (1 / 3) / sigma
        r_clusters = r_clusters[r_clusters > 0]

        r_min = 1e-1
        r_max = 1e1
        bins = np.logspace(np.log10(r_min), np.log10(r_max), 50)

        shape, loc, scale = stats.lognorm.fit(r_clusters)
        mu = np.log(scale)

        result = {
            "p": p,
            "sigma": sigma,
            "r_clusters": r_clusters,
            "bins": bins,
            "shape": shape,
            "loc": loc,
            "scale": scale,
            "mu": mu,
            "grid_dims": grid_dims,
        }

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
            print(f"[Saved] Result cached to {cache_file}")

        return result

    except Exception as e:
        print(f"[Error] Simulation failed for p={p}, dims={grid_dims}: {e}")
        return None

def main_parallel(realizations, grid_dims_list, n_jobs, output_file=None):
    sigma = 8
    tasks = [
        (p, sigma, dims)
        for dims in grid_dims_list
        for p in realizations
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(lognorm_fit_simulation)(p, sigma, dims) for (p, sigma, dims) in tasks
    )

    # Filter out failed
    results = [res for res in results if res is not None]

    # Group results by grid_dims
    from collections import defaultdict
    grouped = defaultdict(list)
    for res in results:
        grouped[res["grid_dims"]].append(res)

    fig, axes = plt.subplots(1, len(grouped), figsize=(8 * len(grouped), 6), sharey=True)

    # Ensure axes is iterable
    if len(grouped) == 1:
        axes = [axes]

    for i, (dims, res_list) in enumerate(grouped.items()):
        ax = axes[i]
        for j, res in enumerate(sorted(res_list, key=lambda x: x['p'], reverse=True)):
            ax.hist(res['r_clusters'], bins=res['bins'], alpha=0.5, label = rf"$f_v = 10^{{{int(np.log10(res['p']))}}}$", color = colours[j%len(colours)])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-1, 1e1)
        ax.set_xlabel(r"$r\ /\ r_{\mathrm{input}}$")
        if i == 0: ax.set_ylabel(r"$N_{\mathrm{clouds}}$")
        #ax.set_title(f"Grid dims: {dims}")
        if i == 1 or len(grouped)==1: ax.legend()

    fig.tight_layout()
    if output_file:
        print("This is the output file:", output_file)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return [(res['p'], res['mu'], res['grid_dims']) for res in results]


# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Percolation Simulation")
    parser.add_argument("--N_procs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()
    n_jobs = args.N_procs

    realizations = [1e-1, 1e-2, 1e-3, 1e-4]
    grid_dims_list = [(800, 800, 800)]

    mu_values = main_parallel(realizations, grid_dims_list, n_jobs, output_file = f'k_estimates_{grid_dims_list[0][0]}.pdf')
    print(mu_values)
