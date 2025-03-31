import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist

def simulate_percolation(dimensions, p, sigma):
    """Simulate percolation with Gaussian smoothing for large 3D fields."""
    field = np.random.rand(*dimensions).astype(np.float32)  # Use float32 for memory efficiency
    if isinstance(sigma, tuple):
        smoothed_field = np.mean(
            [gaussian_filter(field, s, mode='nearest') for s in np.linspace(*sigma, 5)], axis=0)
    else:
        smoothed_field = gaussian_filter(field, sigma=sigma, mode='nearest')

    threshold = np.percentile(smoothed_field, 100 * (1 - p))
    return smoothed_field > threshold

# List of fvs to loop over
fvs = [0.1, 0.01]  # Example values

# Loop over each fv and process
for fv in fvs:
    print(f"Processing fv = {fv}")

    # Simulate percolation field
    density_array = simulate_percolation((200, 200, 200), fv, sigma=0.5)

    # Identify clumps
    threshold = 0.5  # Adjust as needed
    binary_data = (density_array > threshold).astype(int)
    labeled_array, num_features = label(binary_data)

    # Find clump centers
    if num_features > 0:
        clump_positions = np.array(center_of_mass(density_array, labeled_array, index=np.arange(1, num_features + 1)))

        # Compute distances
        print("Computing distances...")
        if len(clump_positions) > 1:
            distances = pdist(clump_positions)
            
            # Plot histogram
            plt.figure()
            plt.hist(distances, bins=30, edgecolor='black')
            plt.xlabel('Separation Distance')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Clump Separations (fv={fv})')
            plt.show()
        else:
            print(f"Only one clump found for fv={fv}, skipping histogram.")
    else:
        print(f"No clumps found for fv={fv}, skipping histogram.")
