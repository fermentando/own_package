import numpy as np
import read_hdf5 as hd
from multiprocessing import Pool
import os
import sys
import matplotlib.pyplot as plt
import argparse
from multiprocessing import cpu_count
from matplotlib.colors import LogNorm

def parallelise(fn, iter_list=None, processes=2):
    """
    Function to distribute a repeated function call over multiple processors

    Args:
        fn (function): Function to be run parallely
        iter_list (list, optional): List with elements to iterate over. Defaults to None.
        processes (int, optional): Number of cores to send the jobs. Defaults to 2.
    """

    if iter_list == None:
        iter_list = [i for i in range(processes)]

    with Pool(processes) as pool:
        processed = pool.map(fn, iter_list)

    return processed

# TODO: Add ray-casting (pin-hole camera) to this
def project_on_plane(points, camera_view):
    # * Ensure points and plane_normal are numpy arrays
    camera_view = np.asarray(camera_view).astype(float)

    points = np.asarray(points).astype(float)

    theta, phi = camera_view

    plane_normal = np.array(
        [
            np.sin(phi),
            np.cos(phi) * np.sin(theta),
            np.cos(phi) * np.cos(theta),
        ]
    )

    # * Ensure plane_normal is a unit vector
    plane_normal /= np.linalg.norm(plane_normal)

    # * Calculate the projection matrix
    projection_matrix = np.eye(3) - np.outer(plane_normal, plane_normal)

    # * Project each point onto the plane
    projected_points = np.matmul(projection_matrix, points.T).T

    # * Create an orthonormal basis for the plane
    # * Choose an arbitrary vector perpendicular to the plane normal
    # * Let plane normal be [a,b,c], and the perp. one be [x, y, z]
    # * Then perpendicular vector follows: a*x + b*y + c*z = 0
    # *  Take x=1, y=1, => a + b + cz = 0 => z = -a/c
    v1 = np.array(
        [
            -np.cos(phi),
            np.sin(phi) * np.sin(theta),
            np.sin(phi) * np.cos(theta),
        ]
    )
    v1 /= np.linalg.norm(v1)

    # * Calculate the second basis vector by taking the cross product
    v2 = np.cross(plane_normal, v1)
    v2 /= np.linalg.norm(v2)

    # * Calculate the third basis vector by taking the cross product of the first two
    v3 = np.cross(v1, v2)

    # * Convert the projected points to the plane coordinate system
    # * plane_coordinates = np.matmul(np.vstack((v1, v2, v3)), projected_points.T).T

    # * Convert the projected points to the plane coordinate system
    plane_coordinates = np.empty((projected_points.shape[0], 2))
    for i in range(projected_points.shape[0]):
        plane_coordinates[i] = np.array(
            [np.dot(v1, projected_points[i]), np.dot(v2, projected_points[i])]
        )

    return plane_coordinates


def project_along_normal(arr, points, camera_view, assign_type="nearest", alpha=None, colormap="viridis"):
    projected_points = project_on_plane(points, camera_view)

    origin = np.min(projected_points, axis=0)
    projected_points -= origin
    projected_index = projected_points.astype(int)

    proj_size_actual = np.max(projected_index, axis=0) + 1
    proj_size = int(np.sqrt(3) * np.max(arr.shape))
    
    offset = ((proj_size - proj_size_actual) / 2).astype(int)
    projected_index += offset
    projected_points += offset

    arr_proj = np.zeros((proj_size, proj_size), dtype=float)
    plot_arr = arr if alpha is None else arr * alpha

    point_indices = tuple(points.T)

    if assign_type == "CIC":
        rounded_proj = np.round(projected_points)
        a = np.abs(rounded_proj - projected_points)
        a_sign = np.sign(rounded_proj - projected_points).astype(int)

        indices_x, indices_y = projected_index.T
        weight = (a[0] + 0.5) * (a[1] + 0.5)

        np.add.at(arr_proj, (indices_x, indices_y), weight * plot_arr[point_indices])
        np.add.at(arr_proj, (indices_x - a_sign[0], indices_y), (0.5 - a[0]) * (a[1] + 0.5) * plot_arr[point_indices])
        np.add.at(arr_proj, (indices_x, indices_y - a_sign[1]), (a[0] + 0.5) * (0.5 - a[1]) * plot_arr[point_indices])
        np.add.at(arr_proj, (indices_x - a_sign[0], indices_y - a_sign[1]), (0.5 - a[0]) * (0.5 - a[1]) * plot_arr[point_indices])
    
    elif assign_type == "nearest":
        np.add.at(arr_proj, (projected_index[:, 0], projected_index[:, 1]), plot_arr[point_indices])

    # Normalize the projected array
    arr_proj_min, arr_proj_max = np.min(arr_proj), np.max(arr_proj)
    norm_arr_proj = (arr_proj - arr_proj_min) / (arr_proj_max - arr_proj_min + 1e-10) if arr_proj_max > arr_proj_min else arr_proj

    return norm_arr_proj

def projection_render(args):
    i, rho_full, alpha, theta_arr, phi = args
    print(f"Processing frame {i}...")

    rho_full = hd.read_hdf5(filename=file_path, fields=["rho", "T"])
    rho_proj = np.copy(rho_full["rho"])
    
    # Apply threshold to avoid issues with log scale (set minimum to small positive value)
    rho_proj[rho_full["T"] > 5e5] = 1e-10  # Ensures no zero values in log scale

    # Project along normal
    rho_proj = project_along_normal(
        arr=rho_proj,  # No need to take log before this step
        points=grid,
        camera_view=[theta_arr[i], phi],
        alpha=alpha,
    )


    # Visualization with LogNorm
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(rho_proj, cmap=cr.torch, norm=LogNorm(vmin=rho_proj.min(), vmax=rho_proj.max()), interpolation='gaussian')

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Log Density Projection")

    ax.set_axis_off()

    # Save image
    plt.savefig(f"rho_projection_{str(i).zfill(5)}.png", dpi=600)
    plt.close(fig)

    return i

# * Transfer function
def alpha_func(x, x_0=0.5, rate=10.0):
    a = x - x.min()
    a /= a.max()
    return 1 / (1 + np.exp(rate * (x_0 - a)))

# Set default number of processes
def get_n_procs():
    parser = argparse.ArgumentParser(description="Set the number of processors.")
    parser.add_argument("N_procs", nargs="?", type=int, default=1, help="Number of processors to use.")
    args = parser.parse_args()
    return max(1, min(args.N_procs, cpu_count()))  # Ensure valid range

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cmasher as cr
    import os
    import sys

    N_procs = get_n_procs()
    print(f"N_procs set to: {N_procs} processors.")

    # Determine package path
    cwd = os.path.abspath(os.path.dirname(__file__))
    package_abs_path = os.path.dirname(cwd)

    # Configuration Flags
    MHD_flag = False
    fixed_time = True

    # File Path
    file_path = "/raven/ptmp/ferhi/ISM_slab/100kc/fv01e/out/parthenon.prim.00007.phdf"

    # Define angles
    phi = np.pi / 4
    theta_arr = np.linspace(2 * np.pi, 0, num=360) if fixed_time else np.linspace((650 - 501) * (np.pi / 180), 0, num=2)

    # Read Data Once
    rho_full = hd.read_hdf5(filename=file_path)
    grid_shape = np.shape(rho_full["rho"])

    # Define 3D grid points
    grid = np.indices(grid_shape).reshape(3, -1).T

    # Prepare for parallel processing
    iter_list = [
        (i, rho_full, alpha_func(rho_full["rho"], x_0=0.4), theta_arr, phi)
        for i in range(len(theta_arr))
    ]

    processed = parallelise(fn=projection_render, iter_list=iter_list, processes=N_procs)