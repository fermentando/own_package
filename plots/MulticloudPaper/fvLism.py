import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import LogNorm

plt.style.use("custom_plot")

class HandlerMultiMarker(HandlerBase):
    def __init__(self, markers, colors, sizes=None, **kwargs):
        HandlerBase.__init__(self, **kwargs)
        self.markers = markers
        self.colors = colors
        self.sizes = sizes if sizes is not None else [8] * len(markers)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        step = width / (len(self.markers) + 1)
        for i, (m, c, s) in enumerate(zip(self.markers, self.colors, self.sizes)):
            x = xdescent + (i+1) * step
            y = ydescent + height/2
            artists.append(Line2D([x], [y], marker=m, color=c,
                                  markersize=s, linestyle="None", transform=trans))
        return artists
# Data
size_r_crit = np.array([10, 10,
                        1, 1, 1, 1, 1, 1,
                        1e-1, 1e-1, 5e-1, 5e-1,
                        1e-2, 5e-2, 1e-2, 
                        10])
fvs = np.array([1e-1, 1e-3, 
                1e-1, 1e-1, 1e-2, 1e-3, 1e-3, 1e-2,
                1e-1, 1e-1, 1e-2, 1e-1, 
                1e-1, 1e-1, 1e-1, 
                1e-2])
LISM = 10*np.array([3, 2,
                 0.3, 3, 20, 30, 400, 2, 
                 30, 3, 4, 4,
                 300, 8, 30,
                 3])
LISM_fv = LISM * fvs #* size_r_crit
print(LISM_fv[11])

# Marker groups
red_cross_indices = np.array([2, 5, 7, 9, 14]) 
green_circle_indices = np.array([0, 1, 3, 4, 6, 8, 12, 15])

orange_cross_indices = np.array([10, 13])  # Example indices for orange crosses
lightblue_circle_indices = np.array([11])  # Example indices for light blue circles

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
sc3 = ax.scatter(size_r_crit[orange_cross_indices], LISM_fv[orange_cross_indices],
                 color='#800080',
                 marker='^', s=100)
sc4 = ax.scatter(size_r_crit[lightblue_circle_indices], LISM_fv[lightblue_circle_indices],
                 color='orange',
                 marker='v', s=80)

# Group into two legend entries


# Plot the question mark markers
#for x, y in zip(question_x, question_y):
#    ax.text(x, y, '?',
#            fontsize=20, fontweight='bold', color='red',
#            ha='center', va='center')



x = np.logspace(-3, np.log10(2),500)
y = 1/x  # y = x line

# Define plot limits
xmin, xmax = 3e-3,5e1
ymin, ymax = 1e-2, 1e3

x1 = np.logspace(np.log10(xmin), np.log10(xmax), 500)
y1 = np.logspace(np.log10(ymin), np.log10(ymax), 500)
X, Y = np.meshgrid(x1, y1)

# Define boundary curve: y = 0.8 / x
boundary = 1 / X

# Mask: above curve OR right of vertical line
mask = (Y >= boundary) | (X >= 2)



# Apply shading
ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['lightblue'], alpha=0.3)

# Fill region below y = x and right of x = 0.5
mask_comp = (Y < boundary) & (X < 2)
ax.contourf(X, Y, mask_comp.astype(int), levels=[0.5, 1], colors=['lightcoral'], alpha=0.3)


# Plot the boundary lines
ax.plot(x, y, 'k--', alpha=0.5, label=r'$\propto r_{crit}$')
ax.vlines(x=2, ymin=0.01, ymax=0.5, colors='k', linestyles='--', alpha=0.5)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Labels and layout
ax.set_xlabel(r'$r_\mathrm{cl} [r_\mathrm{crit}] $')
ax.set_ylabel(r'$fv \ L_\mathrm{ISM} [r_\mathrm{cl}]$')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
combined_handles = [
    (r'$\mathcal{M}_w = 1.5$', dict(markers=['x', 'o'], colors=['red', 'green'])),
    (r'$\mathcal{M}_w = 0.7$', dict(markers=['^', 'v'], colors=['#800080', 'orange']))
]

# Get current legend handles
handles, labels = ax.get_legend_handles_labels()

# Add combined ones
for name, style in combined_handles:
    handles.append(Line2D([], [], linestyle="None"))  # dummy handle
    labels.append(name)

# Draw legend with custom handler
plt.legend(handles, labels, handler_map={
    handles[-2]: HandlerMultiMarker(**combined_handles[0][1]),
    handles[-1]: HandlerMultiMarker(**combined_handles[1][1])
}, loc='best')

print("Figure saved to /u/ferhi/Figures/fvLism_plot.png")
plt.savefig('/u/ferhi/Figures/fvLism_plot.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


