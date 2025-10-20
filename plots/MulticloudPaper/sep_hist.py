import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, gaussian_filter
from scipy.spatial import cKDTree
from cloud_separation import simulate_percolation

def compute_nn_distances(field, sigma, k=6):
    labeled_field, num_clumps = label(field)
    if num_clumps < 2:
        return None

    clump_centroids = np.array(center_of_mass(field, labeled_field, range(1, num_clumps + 1)))
    tree = cKDTree(clump_centroids)
    distances, _ = tree.query(clump_centroids, k=k)
    return distances[:, k-1] / sigma[0]  # Normalize by sigma

if __name__ == "__main__":
    dimensions = (1000, 1000, 1000)
    sigma = [8]
    p_values = [1e-1, 1e-2, 1e-3, 1e-4]  # 4 different p values

    # Smooth field once
    base_field = np.random.normal(0, 1, size=dimensions).astype(np.float32)
    smoothed_fields = [gaussian_filter(base_field, s, mode='nearest') * s**3 for s in sigma]
    smoothed_field = np.mean(smoothed_fields, axis=0) / sum(s**3 for s in sigma)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, p in enumerate(p_values):
        threshold = np.percentile(smoothed_field, 100 * (1 - p))
        binary_field = smoothed_field > threshold
        nn_distances = compute_nn_distances(binary_field, sigma)

        if nn_distances is not None:
            axs[i].hist(nn_distances, bins=30, color='skyblue', edgecolor='black')
            axs[i].set_title(f'p = {p}')
            axs[i].set_xlabel('NN Separation (in units of σ)')
            axs[i].set_ylabel('Frequency')
            axs[i].grid(True)
        else:
            axs[i].text(0.5, 0.5, "Too few clumps", ha='center', va='center')
            axs[i].set_title(f'p = {p}')

    plt.tight_layout()
    plt.savefig(f"histograms_distances_{dimensions[0]}.png")
    plt.show()
