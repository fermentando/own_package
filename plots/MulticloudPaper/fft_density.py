import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, fftshift
import seaborn as sns

plt.figure(figsize=(10, 10))
plt.style.use('custom_plot')
sns.set_palette("crest")

def compute_power_spectrum(density_field):
    """
    Computes the isotropic power spectrum of a 3D density field.
    Only considers non-zero density regions.
    """
    # Mask clump regions
    masked_field = np.where(density_field > 0, density_field, 0)

    # Subtract mean of non-zero regions only
    if np.count_nonzero(masked_field) == 0:
        return None, None  # skip empty realizations

    mean_val = np.mean(masked_field[masked_field > 0])
    mean_subtracted = masked_field - mean_val

    # FFT and power spectrum
    fft_field = fftn(mean_subtracted)
    power = np.abs(fft_field)**2
    power = fftshift(power)

    # Create radial k bins
    size = power.shape[0]
    center = size // 2
    kx, ky, kz = np.meshgrid(np.arange(size) - center,
                             np.arange(size) - center,
                             np.arange(size) - center,
                             indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
    power = power.flatten()

    # Bin isotropically
    k_bins = np.arange(1, size // 2)
    power_spectrum = np.zeros_like(k_bins, dtype=np.float64)

    for i, k in enumerate(k_bins):
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
        if np.any(mask):
            power_spectrum[i] = power[mask].mean()

    return k_bins, power_spectrum

# Load data
percolations = np.load("/viper/u2/ferhi/percolations.npy", allow_pickle=True)

# Initialize plot
for i, percolation in enumerate(percolations):
    print("Percolation no:", i)
    k, Pk = compute_power_spectrum(percolation)
    if k is not None:
        plt.plot(k, Pk, label=f'Realization {i}')

# Reference slopes
# Choose a range inside actual k-values
k_min, k_max = 5, 30  # adjust if needed depending on your data
k_ref = np.linspace(k_min, k_max, 100)

# Use a representative spectrum for vertical scaling
example_idx = len(percolations) // 2
k_sample, P_sample = compute_power_spectrum(percolations[example_idx])
if k_sample is not None:
    idx_ref = np.argmin(np.abs(k_sample - 10))
    y_ref = P_sample[idx_ref]

    # Plot k^-5/3 slope
    slope_53 = y_ref * (k_ref / k_ref[0])**(-5/3)
    plt.plot(k_ref, slope_53, 'k--', label=r'$k^{-5/3}$')

    # Plot k^-7/2 slope
    slope_72 = y_ref * (k_ref / k_ref[0])**(-11/3)
    plt.plot(k_ref, slope_72, 'k-.', label=r'$k^{-11/3}$')

# Final plot adjustments
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$P(k)$')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("density_power_spectrum.png", dpi=300)
plt.show()
