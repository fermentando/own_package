import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.style.use("custom_plot")
# Data
size_r_crit = 1/np.array([10, 10, 10, 10,
                        1, 1, 1, 1, 1,
                        1e-1, 1e-1, 1e-1,
                        1e-2])
fvs = np.array([1e-1, 1e-2, 1e-3, 1e-4,
                1e-1, 1e-2, 1e-2, 1e-2, 1e-2,
                1e-1, 1e-2, 1e-2,
                1e-1])
LISM = np.array([.3, .3, .3, .3,
                 6, 14, 20, 30, 100,
                 60, 60, 700,
                 600])
LISM_fv = LISM * fvs #* size_r_crit
print(LISM_fv[9])

# Marker groups
red_cross_indices = np.array([5, 6, 7, 10]) 
green_circle_indices = np.array([0, 1, 2, 3, 4, 8, 9, 11, 12])

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale('log')
ax.set_yscale('log')


# Plot: normal values (colored)
normal_cross = red_cross_indices
normal_circle = green_circle_indices

sc1 = ax.scatter(size_r_crit[normal_cross], LISM_fv[normal_cross], 
                color='red',
                 marker='x', s=100)
sc2 = ax.scatter(size_r_crit[normal_circle], LISM_fv[normal_circle], 
                 color='green',
                 marker='o', s=80)
ax.plot(np.logspace(-1,2), 0.4*np.logspace(-1,2), 'k--', alpha = 0.5, label=r'$\propto r_{crit}$')

# Labels and layout
ax.set_xlabel(r'$k/k_{crit} $')
ax.set_ylabel(r'$fv \ L_{ism}$')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig('fvLism.png')
plt.show()

