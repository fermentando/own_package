import numpy as np
import read_hdf5 as hd
from multiprocessing import Pool

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


def project_along_normal(arr, points, camera_view, assign_type="nearest", alpha=None):
    # * Project the points onto the plane
    projected_points = project_on_plane(points, camera_view)

    origin = np.min(projected_points, axis=0)

    projected_points -= origin
    projected_index = projected_points.astype(int)

    proj_size_actual = np.max(projected_index, axis=0) + 1
    proj_size = int(np.sqrt(3) * np.max(np.shape(arr)))

    projected_index += ((proj_size - proj_size_actual) / 2).astype(int)
    projected_points += ((proj_size - proj_size_actual) / 2).astype(int).astype(float)

    arr_proj = np.zeros((proj_size, proj_size), dtype=float)

    if alpha is None:
        plot_arr = arr
    else:
        plot_arr = arr * alpha

    if assign_type == "CIC":
        for i in range(np.shape(projected_index)[0]):
            a = np.abs(np.round(projected_points[i, :]) - projected_points[i, :])
            a_sign = np.sign(
                np.round(projected_points[i, :]) - projected_points[i, :]
            ).astype(int)

            arr_proj[tuple(projected_index[i, :])] += (
                (a[0] + 0.5) * (a[1] + 0.5) * plot_arr[tuple(points[i, :])]
            )
            arr_proj[
                tuple([projected_index[i, 0] - a_sign[0], projected_index[i, 1]])
            ] += ((0.5 - a[0]) * (a[1] + 0.5) * plot_arr[tuple(points[i, :])])
            arr_proj[
                tuple([projected_index[i, 0], projected_index[i, 1] - a_sign[1]])
            ] += ((a[0] + 0.5) * (0.5 - a[1]) * plot_arr[tuple(points[i, :])])
            arr_proj[tuple(projected_index[i, :] - a_sign)] += (
                (0.5 - a[0]) * (0.5 - a[1]) * plot_arr[tuple(points[i, :])]
            )
    elif assign_type == "nearest":
        for i in range(np.shape(projected_index)[0]):
            arr_proj[tuple(projected_index[i, :])] += plot_arr[tuple(points[i, :])]

    return arr_proj


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cmasher as cr
    import os
    import sys

    N_procs_default = 1

    n_arg = len(sys.argv) - 1
    if n_arg == 0:
        print(f"N_procs not provided...")
        print(f"N_procs set to default: {N_procs_default} processors..")
        N_procs = N_procs_default
    elif n_arg == 1:
        N_procs = int(sys.argv[1])
        print(f"N_procs set to: {N_procs} processors..")
    else:
        print(f"Too many arguments provided...")
        print(f"N_procs set to default: {N_procs_default} processors..")
        N_procs = N_procs_default

    cwd = os.path.dirname(__file__)
    package_abs_path = cwd[: -len(cwd.split("/")[-1])]

    # *______________________________________________* #
    # *______________________________________________* #

    MHD_flag = False

    fixed_time = True
    
    file_path = '/raven/ptmp/ferhi/ISM_slab/100kc/fv01e/out/parthenon.prim.00007.phdf'

    # *______________________________________________* #
    # *______________________________________________* #

    phi = np.pi / 4
    if fixed_time:
        theta_arr = np.linspace(2 * np.pi, 0, num=360)
    else:
        theta_arr = np.linspace((650 - 501) * (np.pi / 180), 0, num=(650 - 501))


    rho_full0 = hd.read_hdf5(
        filename=file_path
    )

    # * Define some 3D points
    grid_shape = np.shape(rho_full0['rho'])
    grid = np.mgrid[0 : grid_shape[0], 0 : grid_shape[1], 0 : grid_shape[2]]

    points = np.vstack(
        (
            grid[0].ravel(),
            grid[1].ravel(),
            grid[2].ravel(),
        )
    ).T

    rho_full = hd.read_hdf5(
        filename=file_path,
    )

    # def alpha_func(c_arr, log_flag=False):
    #     return p3d.poly_alpha(c_arr, log_flag=log_flag, order=1, cut=5, alpha0=0.75)

    def alpha_func(x, x_0=0.5, rate=10.0):
        a = x - x.min()
        a /= a.max()
        return 1 / (1 + np.exp(rate * (x_0 - a)))

    def projection_render(args):
        i, rho_full, alpha, theta_arr, phi = args
        print(i)
        rho_full = hd.read_hdf5(
            filename=file_path,
            fields = ['rho', 'T']
        )

        rho_proj = np.copy(rho_full['rho'])
        print(rho_proj.max(), rho_proj.min(), flush=True)
        print('This is ratio max/min: ', rho_proj.max()/rho_proj.min(), flush=True)
        rho_proj[rho_full['T'] > 5e5] = 0 #change this??

        rho_proj = project_along_normal(
            arr=np.log10(rho_proj),
            # arr=alpha,
            points=points,
            camera_view=[theta_arr[i], phi],
            alpha=alpha,
        )
        print(rho_proj.max(), rho_proj.min(), flush=True)
        print('This is ratio max/min: ', rho_proj.max()/rho_proj.min(), flush=True)

        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        colormap = cr.torch

        ax.imshow(
            # np.log10(rho_proj),
            rho_proj,
            cmap=colormap,
            interpolation="gaussian",
            # interpolation_stage="rgba",
            alpha=1,
            #vmin = np.sqrt(abs(rho_proj.min()*rho_proj.max())),
            vmin = rho_proj.min()*4,
            #vmax=rho_proj.max(),
        )
        

        ax.grid(False)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])


        if fixed_time:
            plt.savefig(f"New_renderings/rho_projection_{str(i).zfill(5)}.png", dpi=600)

        return i

    iter_list = [
        (
            i,
            rho_full,
            alpha_func(rho_full['rho'], x_0=0.4),
            theta_arr,
            phi,
        )
        for i in range(len(theta_arr))
    ]
    processed = parallelise(
        fn=projection_render,
        iter_list=iter_list,
        processes=N_procs,
    )

    # vid.make_video(
    #     image_path="rho_projection", video_path="rho_proj", framerate=10, theme=theme
    # )