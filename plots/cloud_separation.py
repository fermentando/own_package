import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.ndimage import label, center_of_mass, gaussian_filter, binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import scipy.stats as stats
import os
import pickle
from scipy.optimize import curve_fit
import argparse

plt.style.use("custom_plot")

def generate_clump_field(dimensions, volume_fraction, clump_radius, allow_overlap=True):
    V_box = np.prod(dimensions)
    V_clump = (4 / 3) * np.pi * clump_radius**3
    num_clumps = int(np.round(volume_fraction * V_box / V_clump))

    field = np.zeros(dimensions, dtype=bool)
    rng = np.random.default_rng()
    placed_centers = []

    # Precompute spherical mask
    r = clump_radius
    xg, yg, zg = np.ogrid[-r:r, -r:r, -r:r]
    clump_mask = xg**2 + yg**2 + zg**2 <= r**2

    for _ in range(num_clumps):
        for attempt in range(20):  # Reduced attempts for speed
            x = rng.integers(r, dimensions[0] - r)
            y = rng.integers(r, dimensions[1] - r)
            z = rng.integers(r, dimensions[2] - r)

            if not allow_overlap:
                if placed_centers:
                    dists2 = np.sum((np.array(placed_centers) - [x, y, z])**2, axis=1)
                    if np.min(dists2) < (2 * r)**2:
                        continue  # too close

            # Place clump
            xs = slice(x - r, x + r)
            ys = slice(y - r, y + r)
            zs = slice(z - r, z + r)
            field[xs, ys, zs] |= clump_mask
            placed_centers.append((x, y, z))
            break

    return field

# Simulate percolation function (assuming already defined)
def simulate_percolation(dimensions, p, sigmas):
    """Simulate percolation with Gaussian smoothing for large 3D fields."""
    field = np.random.normal(0,1,size=dimensions).astype(np.float32)
    fields = []
    for sigma in sigmas:
        smoothed_field = gaussian_filter(field, sigma=sigma, mode='nearest')* sigma ** 3
        fields.append(smoothed_field)
    smoothed_field = np.mean(np.array(fields), axis=0)/ sum(sigma**3 for sigma in sigmas)
    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    return smoothed_field > threshold

# Function to compute nearest-neighbor distance for percolation field (assuming already defined)
def compute_nn_distance_percolation(binary_field, k=6):
    """Compute the average nearest-neighbor distance between different clumps in a 3D percolation field."""
    
    #nearest_distances = []

    field = binary_field
    labeled_field, num_clumps = label(field)
    #sizes = np.bincount(labeled_field.ravel())[1:]  # skip background
    #avg_size = np.percentile(sizes,25)
    
    if num_clumps < 2:
        return None 
    
    #large_clump_ids = np.where(sizes > avg_size)[0] + 1 
    clump_centroids = np.array(center_of_mass(field, labeled_field, range(1, num_clumps + 1)))

    tree = cKDTree(clump_centroids)
    distances, _ = tree.query(clump_centroids, k=k)  # k=2 to exclude self
    
    #nearest_distances.extend(distances[:, 1])  # Exclude self-distance
    nearest_distances = distances[:,k-1]
    median = np.median(nearest_distances)
    lower = np.percentile(nearest_distances, 16)
    upper = np.percentile(nearest_distances, 84)

    # Print for debugging
    print(f"Median NN distance: {median:.3f}, 16th: {lower:.3f}, 84th: {upper:.3f}")

    # Return median and symmetric error (can also return asymmetrical if needed)
    error = max(median - lower, upper - median)
    return median, lower, upper
    #else:
    #    return None

def compute_voxel_distance(clump1_mask, clump2_mask):
    coords1 = np.argwhere(clump1_mask)
    coords2 = np.argwhere(clump2_mask)
    dists = cdist(coords1, coords2)
    return dists.min()

def compute_nn_distance_fast(binary_field, distance_threshold=8, max_checks=20):
    field = binary_field
    labeled_field, num_clumps = label(field)

    if num_clumps < 2:
        return None

    clump_centroids = np.array(center_of_mass(field, labeled_field, range(1, num_clumps + 1)))
    tree = cKDTree(clump_centroids)

    nearest_distances = []

    for i, centroid in enumerate(clump_centroids):
        clump_id = i + 1
        mask1 = (labeled_field == clump_id)

        distances, indices = tree.query(centroid, k=min(max_checks + 1, num_clumps))

        for dist, idx in zip(distances[1:], indices[1:]):  # Skip self
            neighbor_id = idx + 1
            mask2 = (labeled_field == neighbor_id)

            min_voxel_dist = compute_voxel_distance(mask1, mask2)
            if min_voxel_dist > distance_threshold:
                nearest_distances.append(min_voxel_dist)
                break  # Found acceptable neighbor

    nearest_distances = np.array(nearest_distances)
    if len(nearest_distances) == 0:
        return None

    median = np.mean(nearest_distances)
    lower = np.percentile(nearest_distances, 16)
    upper = np.percentile(nearest_distances, 84)

    print(f"KD-tree w/ voxel filter: Median {median:.3f}, 16th {lower:.3f}, 84th {upper:.3f}")
    return median, lower, upper


def straight_line(x, m, x0, y0):
    C = y0 - m*x0
    return m*x + C

def power_law_model(x, A, B):
    return A * x**B

def compute_distances_for_dimension(dimensions, p_values, sigmas):
    mean_distances = []
    std_distances = []
    upper = []

    # Smooth only once
    base_field = np.random.normal(0, 1, size=dimensions).astype(np.float32)
    smoothed_fields = [gaussian_filter(base_field, sigma=sigma[0], mode='nearest')]# * s ** 3 for s in sigmas]
    smoothed_field = np.mean(smoothed_fields, axis=0)# / sum(s**3 for s in sigmas)

    for p in p_values:
        threshold = np.percentile(smoothed_field, 100 * (1 - p))
        binary_field = smoothed_field > threshold

        result = compute_nn_distance_percolation(binary_field)

        if result is not None:
            mean_distances.append(result[0])
            std_distances.append(result[1])
            upper.append(result[2])
        else:
            mean_distances.append(0)
            std_distances.append(0)
            upper.append(0)

    return np.array(mean_distances), np.array(std_distances), np.array(upper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run percolation simulations.")
    parser.add_argument("--N_procs", type=int, default=1, help="Number of parallel jobs (e.g., 1 for serial, -1 for all cores)")
    args = parser.parse_args()

    n_jobs = args.N_procs
    # Dimensions list (modify to include the different dimensions)
    dimensions_list = [(1000,1000,1000)]#, (300, 300, 320)]#, (300, 300, 384)]
    p_values = np.logspace(-6, -1, 12)  # Log-spaced p values
    sigma = [8]#np.linspace(0.5,8,5)

    results_file = "percolation_results.pkl"

    # Try to load cached results
    if False:# os.path.exists(results_file):
        with open(results_file, "rb") as f:
            all_mean_distances, all_std_distances = pickle.load(f)
        print("Loaded cached results.")
    if True:
        # Compute results
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_distances_for_dimension)(dim, p_values, sigma) for dim in dimensions_list
        )
        print("Working")
        all_mean_distances = {}
        all_std_distances = {}
        all_upper = {}

        for i, dim in enumerate(dimensions_list):
            mean_distances, std_distances, upper = results[i]
            all_mean_distances[dim] = mean_distances
            all_std_distances[dim] = std_distances
            all_upper[dim] = upper

        # Save results
        #with open(results_file, "wb") as f:
        #    pickle.dump((all_mean_distances, all_std_distances), f)
        print("Saved computed results.")

    # Plotting
    plt.figure(figsize=(8, 6))

    colors = ['blue', 'green', 'red']  # Adjust colors as needed
    dmin = sigma[0]
    # Loop over the results of each dimension
    for i, dimensions in enumerate(dimensions_list):
        mean_distances = all_mean_distances[dimensions]
        std_distances = all_std_distances[dimensions]
        
        # Normalize by the minimum mean distance
        if dmin == None:
            dmin = min(mean_distances)
            dminstd = std_distances[-1]
        new_mean_distances = mean_distances / dmin
        std_distances_normalized = new_mean_distances / dmin
        upper /= dmin


        # Plot the mean distance with error region
        plt.plot(p_values, new_mean_distances, 'o-', color=colors[i])
        plt.fill_between(p_values, new_mean_distances - std_distances_normalized, new_mean_distances + upper, alpha=0.3, color=colors[i])
    plt.plot(p_values,  1 * p_values **(-1/3), linestyle='--', color = "black", label = r"$ d \propto {f_v}^{-1/3}$")
    #plt.plot(p_values,  1 * p_values **(-0.2), linestyle='--')
    #plt.plot(p_values,  1 * p_values **(-0.1), linestyle='--')

        
    # Final plot settings
    plt.xlabel(r'$f_v$')
    plt.ylabel(r'$d [r_{\rm cloud}]$')
    plt.xscale('log')
    plt.yscale("log")
    plt.ylim(bottom = 1)
    plt.legend()
    plt.grid()
    plt.savefig(f'cloud_separation_8rcl_{dimensions[0]}_.png')
    plt.show()

