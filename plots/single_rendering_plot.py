import numpy as np
import matplotlib.pyplot as plt
import os
from read_hdf5 import read_hdf5
import pyvista as pv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import cmcrameri.cm as cm

cmap = cm.roma 
spectral = sns.color_palette("flare", as_cmap=True)
colors = [
    (0.4, 0.7, 1.0),     # stronger light blue (less white)
    (0.0, 0.2, 0.5),     # deep blue
    (0.0, 0.0, 0.0),     # black
    (0.5, 0.2, 0.0),     # dark orange
    (1.0, 0.8, 0.2)      # bright yellow
]

custom_cmap = LinearSegmentedColormap.from_list("stronger_blue_black_orange_yellow", colors)


plt.style.use("custom_plot")
def detect_cold_box(temp, threshold=1e4, padding=5):
    cold_mask = temp < threshold
    coords = np.argwhere(cold_mask)
    if coords.size == 0:
        raise ValueError("No cold gas found.")
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    shape = temp.shape
    return (
        (max(zmin - padding, 0), min(zmax + padding, shape[0])),
        (max(ymin - padding, 0), min(ymax + padding, shape[1])),
        (max(xmin - padding, 0), min(xmax + padding, shape[2]))
    )

# File paths
file_paths = [
    '/viper/ptmp2/ferhi/d3rcrit/10kc/fv01/out/parthenon.prim.00006.phdf',
    #'/viper/ptmp2/ferhi/d3rcrit/kc/fv01_v2/out/parthenon.prim.00000.phdf',
    #'/viper/ptmp2/ferhi/d3rcrit/01kc/fv01/out/parthenon.prim.00000.phdf'
]

print("Detecting cold gas region...")
data0 = read_hdf5(file_paths[0], fields=['T', 'rho'], n_jobs=4)
zrange, yrange, xrange_ = detect_cold_box(data0['T'], threshold=3e4, padding=5)

rho = data0['rho'][xrange_[0]:xrange_[1], yrange[0]-100:yrange[1]//2 - 100 , zrange[0]:zrange[1]]
xmin, xmax = 0, rho.shape[0]
ymin, ymax = 0, rho.shape[1]
zmin, zmax = 0, rho.shape[2]
centre = tuple([s // 2 for s in rho.shape])
cn1, cn2, cn3 = centre
eye0 = (cn1 - 2 * rho.shape[2], cn2 - 10 * rho.shape[2], cn3 - 20 * rho.shape[2])
bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

opacity = [0.05, 0.05, 0.05, 0.1, 0.5, 0.8]  # Automatically spread over scalar range
# Define colors as RGB tuples (normalized 0 to 1)
metallic_yellow = (1.0, 0.9, 0.4)  # light golden yellow
metallic_blue = (0.2, 0.4, 0.6)     # cool silvery blue

metallic_cmap = LinearSegmentedColormap.from_list(
    "metallic_yellow_blue",
    [metallic_yellow, metallic_blue]
)

pl = pv.Plotter(off_screen=True)
box = pv.Box(bounds=bounds)
pl.add_volume(np.log10(rho), scalars="values", cmap=custom_cmap, clim=[-26.4, -24.4], opacity=opacity, show_scalar_bar=False)
pl.add_mesh(box, color="white", style="wireframe", line_width=2)
pl.camera_position = [eye0, (cn1 , cn2 , cn3+ 2 * rho.shape[2]), (1, 0, 0)]
pl.screenshot("long_box.png", transparent_background=True, window_size=[7680, 4320])

from PIL import Image

img = Image.open("long_box.png")
img = img.crop(img.getbbox())  # Automatically trims transparent edges
img.save("long_box.png")

