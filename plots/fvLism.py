import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.style.use("custom_plot")
# Data
size_r_crit = np.array([10, 10,
                        1, 1, 1, 1, 1, 1,
                        1e-1, 1e-1, 2e-1, 2e-1,
                        1e-2, 2e-2])
fvs = np.array([1e-1, 1e-3,
                1e-1, 1e-1, 1e-2, 1e-3, 1e-3, 1e-2,
                1e-1, 1e-1, 1e-2, 1e-1, 
                1e-1, 1e-1])
LISM = 10*np.array([3, 2,
                 0.3, 3, 20, 30, 400, 2, 
                 30, 3, 4, 4,
                 300, 8])
LISM_fv = LISM * fvs #* size_r_crit
print(LISM_fv[11])

# Marker groups
red_cross_indices = np.array([2, 5, 7, 9, 10, 13]) 
green_circle_indices = np.array([0, 1, 3, 4, 6, 8, 11])
questionable_indices = np.array([12])

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
question_x = size_r_crit[questionable_indices]
question_y = LISM_fv[questionable_indices]

# Plot the question mark markers
for x, y in zip(question_x, question_y):
    ax.text(x, y, '?',
            fontsize=20, fontweight='bold', color='red',
            ha='center', va='center')



x = np.logspace(-3, np.log10(2),500)
y = 0.8/x  # y = x line

# Define plot limits
xmin, xmax = 1e-3,5e1
ymin, ymax = 1e-2, 1e3

x1 = np.logspace(np.log10(xmin), np.log10(xmax), 500)
y1 = np.logspace(np.log10(ymin), np.log10(ymax), 500)
X, Y = np.meshgrid(x1, y1)

# Define boundary curve: y = 0.8 / x
boundary = 0.8 / X

# Mask: above curve OR right of vertical line
mask = (Y >= boundary) | (X >= 2)



# Apply shading
ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['lightblue'], alpha=0.3)

# Fill region below y = x and right of x = 0.5
mask_comp = (Y < boundary) & (X < 2)
ax.contourf(X, Y, mask_comp.astype(int), levels=[0.5, 1], colors=['lightcoral'], alpha=0.3)


# Plot the boundary lines
ax.plot(x, y, 'k--', alpha=0.5, label=r'$\propto r_{crit}$')
ax.vlines(x=2, ymin=0.01, ymax=0.3, colors='k', linestyles='--', alpha=0.5)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Labels and layout
ax.set_xlabel(r'$r [r_{crit}] $')
ax.set_ylabel(r'$fv \ L_\mathrm{ISM} [r_\mathrm{cl}]$')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc="upper right")
plt.savefig('/u/ferhi/Figures/fvLism_plot.png', dpi=300, bbox_inches='tight')
plt.show()


