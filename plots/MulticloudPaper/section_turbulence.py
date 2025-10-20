import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
from plot_2d_image import plot_projection
import matplotlib.gridspec as gridspec
from turb_y import hdf_turb_vs_y
from read_hdf5 import read_hdf5
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MultipleLocator
from adjust_ics import SingleCloudCC



filepath = '/viper/ptmp/ferhi/fvLism/01kc/fv01_30r/'
snapshot = 30
colours = ["#d62728", "#1f77b4"]

cmap_red = sns.color_palette("light:#d62728", as_cmap=True)
cmap_blue = sns.color_palette("light:#1f77b4", as_cmap=True)
norm_red = Normalize(vmin=0, vmax=3)
norm_blue = Normalize(vmin=0, vmax=1.5)

# Set up figure and gridspec
plt.style.use("custom_plot")
fig = plt.figure(figsize=(8, 7))
gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[1.5,2,2,0.4], hspace=0.1)

# Secondary plot (turbulence) on top
ax_profile = fig.add_subplot(gs[0])
ax_main_cold = fig.add_subplot(gs[1], sharex=ax_profile)
ax_main_hot = fig.add_subplot(gs[2], sharex=ax_profile)
for j, mode in enumerate(['hot', 'cold']):
    ycords, v_normalised = hdf_turb_vs_y(filepath, mode=mode, snapshot_index=snapshot)
    ax_profile.plot(ycords/240, v_normalised/14e5, color = colours[j], label=mode)

ax_profile.set_ylabel(r'$ v_{turb} / c_{\mathrm{s, cold}}$', labelpad = 20)
ax_profile.set_ylim(0,3)
ax_profile.legend(loc="upper right")


#Load projection
run = os.path.join(filepath, 'out/parthenon.prim.'+str(snapshot).zfill(5)+'.phdf')
print(run)
data = read_hdf5(run, ['rho', 'vel1', 'vel3', 'T'], n_jobs=4)

rho = data['rho']
vel1 = data['vel1']
vel3 = data['vel3']
T = data['T']

#Compute mesh turbulence
vt = np.sqrt(3/2*vel1**2 + 3/2*vel3**2)/14e5
vt_cold = vt.copy()
rho_cold = rho.copy()
rho_cold[T > 1e5] = 0
vt_cold[T > 1e5] = 0 


#Project density
view_dir = 2
L = np.shape(rho)
dim = len(L)

x_dir = (view_dir + 1) % dim
y_dir = (view_dir + 2) % dim
z_dir = view_dir


x_data = np.linspace(0, L[x_dir]/240, num=L[x_dir] + 1)
y_data = np.linspace(0, L[y_dir]/240, num=L[y_dir] + 1)
z_data = np.linspace(0, L[z_dir]/240, num=L[z_dir] + 1)
weight_data = np.ones_like(rho)

rho_proj = np.sum(rho * weight_data, axis=view_dir) / np.sum(weight_data, axis=view_dir)

# Main plot (density projection)
plot_dict_cold = plot_projection(
    vt_cold,
    weight_data=rho_cold,
    view_dir=2,
    cmap=cmap_blue,
    rescale = 240,
    new_fig=False,
    cbar_flag=False,
    fig=fig,
    ax=ax_main_cold,
    kwargs={'norm': norm_blue}
)

ax_main_cold.set_ylabel(r'$y / L_{\mathrm{ISM}}$')
#ax_main_cold.set_yticklabels([])
ax_main_cold.grid(False)



contour_levels = [1e-26, 1e-25, 7e-25, 9e-25, 1e-24]
x_centers = 0.5 * (x_data[:-1] + x_data[1:])
y_centers = 0.5 * (y_data[:-1] + y_data[1:])
X, Y = np.meshgrid(y_centers, x_centers) 
contour = ax_main_cold.contour(
    X, Y,
    rho_proj,  # transpose to match (ny, nx)
    levels=contour_levels,
    colors='black',
    norm=LogNorm(),
    linewidths=0.7,
    alpha=0.4
)

vt_hot = vt.copy()
vt_hot[T < 1e5] = 0 
rho_hot = rho.copy()
rho_hot[T < 1e5] = 0

plot_dict_hot = plot_projection(
    vt_hot,
    weight_data=rho_hot,
    view_dir=2,
    cmap=cmap_red,
    rescale = 240, 
    new_fig=False,
    cbar_flag=False,
    fig=fig,
    ax=ax_main_hot,
    kwargs={'norm': norm_red}

)

tick_label_pad = 7 

# Apply to all axes
for ax in [ax_profile, ax_main_cold, ax_main_hot]:
    ax.tick_params(axis='x', pad=tick_label_pad)
    ax.tick_params(axis='y', pad=tick_label_pad)

ax_profile.label_outer()
ax_main_cold.label_outer()
ax_main_hot.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
ax_main_hot.xaxis.set_visible(True)
ax_main_hot.xaxis.set_tick_params(labelbottom=True)

ax_main_hot.set_xlabel(r'$x / L_{\mathrm{ISM}}$')
ax_main_hot.set_ylabel(r'$y / L_{\mathrm{ISM}}$')
ax_main_hot.tick_params(axis='x', labelbottom=True) 
ax_main_hot.grid(False)





cax1 = fig.add_axes([0.15, 0.05, 0.3, 0.02])  # [left, bottom, width, height]
cax2 = fig.add_axes([0.55, 0.05, 0.3, 0.02])

# Create colorbars
cb1 = fig.colorbar(plot_dict_cold['slc'], cax=cax1, orientation='horizontal')
cb2 = fig.colorbar(plot_dict_hot['slc'], cax=cax2, orientation='horizontal')

cb1.set_label(r'$ v_\mathrm{turb, cold} / c_{\mathrm{s, cold}}$')
cb2.set_label(r'$ v_\mathrm{turb, hot} / c_{\mathrm{s, cold}}$')
    
plt.savefig('example_section.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

"""
for axs in [ax_main_cold, ax_main_hot]:
    for spine in axs.spines.values():
        spine.set_color('white')

    axs.xaxis.set_tick_params(color='white')
    axs.yaxis.set_tick_params(color='white')
"""