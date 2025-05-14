import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, gaussian_filter
import seaborn as sns
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatter


def simulate_percolation(dimensions, p, sigma):
    """ Simulate percolation with Gaussian smoothing. """
    field = np.random.rand(*dimensions)
    smoothed_field = gaussian_filter(field, sigma=sigma) if not isinstance(sigma, tuple) else np.mean(
        [gaussian_filter(field, s) for s in np.linspace(*sigma, 5)], axis=0)
    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    return smoothed_field > threshold


def compute_p_values(sigmas, p_init):
    """ Compute p values for different sigmas. """
    sigmas_new = sorted(sigmas.copy())
    s0 = min(sigmas)
    p_values = []

    for sigma in sigmas_new:
        print("sigma:", sigma)
        p_values.append(p_init * (s0 / sigma) ** 3.)
    return p_values, sigmas_new


def clump_cumulative_distribution(binary_field):
    """
    Given a binary field (e.g., final_percolation), return the cumulative number of clumps (N(>V))
    for clump sizes equal to or above the value.
    """
    labeled_array, _ = label(binary_field)
    clump_sizes = np.bincount(labeled_array.ravel())[1:]  # exclude background
    sorted_sizes = np.sort(clump_sizes)[::-1]
    cumulative_counts = np.arange(1, len(sorted_sizes) + 1)  # cumulative number of clumps
    return sorted_sizes, cumulative_counts

# Creating percolation fields for each lengthscale
if False:
    sigmas = list(np.linspace(8, 100, 40))
    p_values, sprime = compute_p_values(sigmas, 0.1)
    print("p_values:", p_values)
    print("sigmas:", sprime)

    percolations = []
    dimension_field = [(300, 300, 300), (200, 200, 200), (400, 400, 400)]
    for dimensions in dimension_field:
        buffers = []

        for i, sigma in enumerate(sprime):
            p = p_values[i]
            print("no. p:", i, p)
            if p < 0.001:
                break
            buffer = simulate_percolation(dimensions, p, sigma)
            buffers.append(buffer)
        percolations.append(np.sum(buffers, axis=0) > 0)
    np.save("percolations.npy", percolations)


if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    plt.style.use('custom_plot')
    sns.set_palette("crest")
    # Loop through each percolation realization
    percolations = np.load("/viper/u2/ferhi/percolations.npy", allow_pickle=True)
    for i in range(len(percolations)):
        percolation = percolations[i]
        print("Percolation no :", i)
        volumes, _ = clump_cumulative_distribution(percolation)
        volumes = volumes[np.isfinite(volumes) & (volumes > 0)]/8**3

        sorted_volumes = np.sort(volumes)

        # Define common log-spaced bins across all datasets
        bins = np.logspace(np.log10(min(sorted_volumes)), np.log10(max(sorted_volumes)), num=30)

        # Compute CCDF: count clumps ≥ each bin edge
        ccdf_counts = [np.sum(sorted_volumes >= b) for b in bins]

        # Plot as histogram-style step
        plt.step(bins, ccdf_counts, where='post')#, label=r'$Box$ ' + f'{dimension_field[0][0]/8:g}' + r' $r_{cl, min}$')


    # Optional: overlay a fitted model from one realization if needed
    # plt.plot(v, power_law(v, A_est), 'r--', label='Fitted Power-law')
    # plt.fill_between(v, power_law(v, A_lower), power_law(v, A_upper), color='gray', alpha=0.3)

    plt.plot(volumes, 10**3 * volumes.astype(float)**-0.8, 'r--', label=r'$ N \propto V^{-1}$')
    plt.fill_between(bins, 1e6, 1, where=bins < 1,
                    color='gray', alpha=0.3, interpolate=True)
    plt.xticks([1e-2, 1e0, 1e2, 1e4], [r'$10^{-2}$', r'$10^{0}$', r'$10^{2}$', r'$10^{4}$'])





    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$V \ [ r^3_\mathrm {cl} ]$')
    plt.ylim(bottom=1, top = max(ccdf_counts) * 1.5)
    plt.ylabel(r'$ N (\geq V)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("dndm.png", dpi=300)
    plt.show()