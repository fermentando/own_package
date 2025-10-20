import numpy as np
import cupy as cp  # GPU-accelerated NumPy
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit, prange
from joblib import Parallel, delayed
import os
import utils as ut

print('Hello world...')
USE_GPU = True  # Set to False for CPU-only processing

def read_hdf5(filename=None, fields=['rho']):
    if filename is None:
        filename = "/raven/ptmp/ferhi/ISM_slab/100kc/fv01e/out/parthenon.prim.00007.phdf"

    reader = ut.AthenaPKInputFileReader(os.path.abspath(os.path.join(filename, "../../ism.in")))
    code_units_rho = float(reader.get('units', 'code_mass_cgs')) / float(reader.get('units', 'code_length_cgs'))**3

    with h5py.File(filename, "r") as f:
        prim = f["prim"][()]
        rho = prim[:, 0, :, :, :] * code_units_rho

    return {'rho': rho}

file_path = "/raven/ptmp/ferhi/ISM_slab/100kc/fv01e/out/parthenon.prim.00007.phdf"
rho_full = read_hdf5(filename=file_path, fields=['rho'])


if USE_GPU:
    rho_full['rho'] = cp.asarray(rho_full['rho'])


grid_shape = np.shape(rho_full['rho'])
grid = np.mgrid[0:grid_shape[0], 0:grid_shape[1], 0:grid_shape[2]]
points = np.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).T

phi = np.pi / 4
theta_arr = np.linspace(2 * np.pi, 0, num=2)


def alpha_func(density, x_0=0.5, rate=10.0):
    density = np.clip(density, 1e-30, None) 
    alpha_values = np.zeros_like(density, dtype=np.float32)
    chunk_size = 16

    for i in range(0, density.shape[0], chunk_size):
        chunk = density[i:i+chunk_size]
        alpha_values[i:i+chunk_size] = 1 / (1 + np.exp(rate * (x_0 - np.log10(chunk))))
    
    return alpha_values


@njit(parallel=True, fastmath=True)
def project_on_plane(points, theta, phi):
    """Applies 3D-to-2D projection using rotation matrices."""
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
        [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
    ])
    return np.dot(points, R.T[:, :2])  # 3D → 2D Projection


def projection_render(i, density, alpha, theta_arr, phi):
    print(f"🚀 Rendering Frame {i}...")


    theta = theta_arr[i]
    projected_points = project_on_plane(points, theta, phi)
    projected_index = np.round(projected_points - np.min(projected_points, axis=0)).astype(int)

    proj_size = np.max(projected_index, axis=0) + 1
    arr_proj = cp.zeros((proj_size[0], proj_size[1])) if USE_GPU else np.zeros((proj_size[0], proj_size[1]))

    cp.add.at(arr_proj, (projected_index[:, 0], projected_index[:, 1]), density.ravel()) if USE_GPU else np.add.at(arr_proj, (projected_index[:, 0], projected_index[:, 1]), density.ravel())

    # Convert back to NumPy for plotting if using GPU
    if USE_GPU:
        arr_proj = arr_proj.get()

    #  Apply Log Normalization
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(arr_proj, cmap="magma", norm=LogNorm(vmin=arr_proj[arr_proj > 0].min(), vmax=arr_proj.max()))

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Log Density Projection")

    ax.set_axis_off()
    plt.savefig(f"New_renderings/rho_projection_{str(i).zfill(5)}.png", dpi=300)
    plt.close(fig)

    return i

iter_list = [(i, rho_full['rho'], alpha_func(rho_full['rho']), theta_arr, phi) for i in range(len(theta_arr))]
N_procs = os.cpu_count()

Parallel(n_jobs=N_procs)(delayed(projection_render)(*args) for args in iter_list)

print("🚀 Rendering Complete! Frames saved as rho_projection_XXXXX.png")



