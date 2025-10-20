import numpy as np
import matplotlib.pyplot as plt

# Constants
BOX_SIZE = (100,100,100)  # Shape of the 3D domain
CENTER = np.array(BOX_SIZE) // 2
CELL_MASS = 1e-24  # g/cm^3, uniform density
E_TOTAL = 1e51  # erg, total energy injected
E_THERMAL_FRAC = 0.7
E_THERMAL = E_TOTAL * E_THERMAL_FRAC
E_KINETIC = E_TOTAL * (1 - E_THERMAL_FRAC)
M_SPHERE = 3 * 1.989e33  # 3 solar masses in grams
N_CENTRAL = 8  # Central 8 cells (2x2x2 cube)

# Initialize the array: shape (nx, ny, nz, 5), with fields: rho, px, py, pz, energy
data = np.zeros(BOX_SIZE + (5,))
data[..., 0] = CELL_MASS  # Uniform density

# Coordinates
x = np.arange(BOX_SIZE[0])
y = np.arange(BOX_SIZE[1])
z = np.arange(BOX_SIZE[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
positions = np.stack((X, Y, Z), axis=-1)

# Find central region (2x2x2 cube)
cx, cy, cz = CENTER
half = N_CENTRAL // 2
r = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
mask = r <= half
print(len(mask))

# Calculate volume of each cell (arbitrary units since we don’t have dx)
cell_volume = 1.0  # Assume unit volume for simplicity
num_central_cells = np.sum(mask)
mass_per_cell = M_SPHERE / num_central_cells
data[mask, 0] = mass_per_cell / cell_volume  # Adjust density in the central region

# Energy per central cell
thermal_energy = E_THERMAL / num_central_cells
kinetic_energy = E_KINETIC / num_central_cells

# Set thermal energy
data[mask, 4] = thermal_energy

# Assign radial momentum for kinetic energy
positions_center = positions[mask] - CENTER
r = np.linalg.norm(positions_center, axis=1)
r[r == 0] = 1e-10  # Avoid division by zero
directions = positions_center / r[:, np.newaxis]

# Calculate velocity magnitude from kinetic energy
vel_mag = np.sqrt(2 * kinetic_energy / (mass_per_cell))

# Convert to momentum
momentum = vel_mag * mass_per_cell * directions
data[mask, 1:4] = momentum

# Plotting a central slice (z = center)
slice_z = CENTER[2]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Density slice
im0 = axs[0].imshow(data[:, :, slice_z, 0], origin='lower', cmap='viridis')
axs[0].set_title('Density Slice (g/cm³)')
plt.colorbar(im0, ax=axs[0])

# Energy slice
im1 = axs[1].imshow(data[:, :, slice_z, 4], origin='lower', cmap='inferno')
axs[1].set_title('Energy Slice (erg)')
plt.colorbar(im1, ax=axs[1])

# Momentum magnitude slice
momentum_mag = np.linalg.norm(data[:, :, slice_z, 1:4], axis=-1)
im2 = axs[2].imshow(momentum_mag, origin='lower', cmap='plasma')
axs[2].set_title('Momentum Magnitude Slice (g·cm/s)')
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()
