import numpy as np
import pyvista as pv
from read_hdf5 import read_hdf5
import glob
import imageio
import os
from utils import get_n_procs_and_user_args
from joblib import Parallel, delayed



def find_interface_and_extract_prism(rho, nx1, nx2, nx3, threshold=1e-25):
    """
    Vectorized version to detect dense gas interface along y-axis and extract centered prism.

    Parameters:
        rho (ndarray): 3D array (z, y, x) of density values.
        nx1, nx2, nx3 (int): Desired prism size in (z, y, x) directions.
        threshold (float): Density threshold to define 'dense' gas.

    Returns:
        prism (ndarray): Subvolume centered around detected interface.
    """
    zmax, ymax, xmax = rho.shape

    # Create a boolean mask where rho > threshold
    mask = rho > threshold  # shape (z, y, x)

    # Find the first y-index along axis=1 (y) where mask is True
    # Fill invalids with ymax
    first_hits = mask.argmax(axis=1)
    no_hits = ~mask.any(axis=1)
    first_hits[no_hits] = -1  # Mark columns where no dense gas was found

    # Filter out invalid hits
    valid_hits = first_hits[first_hits >= 0]
    if valid_hits.size == 0:
        raise ValueError("No dense interface found in the volume.")

    yc = int(np.median(valid_hits))  # Median y-index of interface
    zc, xc = zmax // 2, xmax // 2    # Center in z and x

    # Compute bounds
    z1 = max(zc - nx1 // 2, 0)
    z2 = min(z1 + nx1, zmax)

    y1 = max(yc - nx2 // 2, 0)
    y2 = min(y1 + nx2, ymax)

    x1 = max(xc - nx3 // 2, 0)
    x2 = min(x1 + nx3, xmax)

    prism = rho[z1:z2, y1:y2, x1:x2]

    return prism



def render_frame_parallel(file_info, hold_frames, rot_frames, spiral_frames, thetas, phi, r_vals, output_dir):
    file_path, globalindx = file_info
    frame_path = os.path.join(output_dir, f"frame_{globalindx:03d}.png")
    
    if os.path.exists(frame_path):
        print(f"Skipping existing frame: {frame_path}")
        return

    rho_read = read_hdf5(file_path, n_jobs=4)['rho']
    rho = find_interface_and_extract_prism(rho_read, rho_read.shape[0], 768, rho_read.shape[2])
    xmin, xmax = 0, rho.shape[0]
    ymin, ymax = 0, rho.shape[1]
    zmin, zmax = 0, rho.shape[2]
    centre = tuple([s // 2 for s in rho.shape])
    cn1, cn2, cn3 = centre
    eye0 = (cn1, cn2, cn3 - 10 * rho.shape[2])
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

    pl = pv.Plotter(off_screen=True)
    box = pv.Box(bounds=bounds)
    pl.add_volume(rho, scalars="values", cmap="plasma", clim=[1e-26, 1e-24], show_scalar_bar=False)
    pl.add_mesh(box, color="white", style="wireframe", line_width=0.1)
    pl.set_background((0.05, 0.05, 0.05))

    if file_path in hold_frames:
        pl.camera_position = [eye0, centre, (1, 0, 0)]

    elif file_path in rot_frames:
        indx = rot_frames.tolist().index(file_path)
        new_vector = (np.cos(thetas[indx]), np.sin(thetas[indx]), 0)
        pl.camera_position = (eye0, centre, new_vector)

    elif file_path in spiral_frames:
        indx = spiral_frames.tolist().index(file_path)
        z = cn3 - r_vals[indx] * rho.shape[2] * np.cos(phi[indx])
        x = cn1 + r_vals[indx] * rho.shape[2] * np.sin(phi[indx])
        eye = (x, cn2, z)
        new_vector = (0, 1, 0)
        pl.camera_position = (eye, centre, new_vector)

    pl.screenshot(frame_path)
    print(f"Saved: {frame_path}")

def rot_and_spiral_zoom_in(files, nhold=0.2, nrot=0.2, nspiral=0.7, output_dir=None, n_workers=4):
    files = np.sort(files)
    n_total = len(files)
    n_hold = int(n_total * nhold)
    n_rot = int(n_total * nrot)
    n_spiral = int(n_total * nspiral)

    hold_frames = files[:n_hold]
    rot_frames = files[n_hold:n_hold + n_rot]
    spiral_frames = files[n_hold + n_rot:n_hold + n_rot + n_spiral]

    thetas = np.linspace(0, np.pi / 2, n_rot)
    phi = np.linspace(0, np.pi / 2, n_spiral)
    r_vals = np.linspace(10, 5, n_spiral)

    os.makedirs(output_dir, exist_ok=True)

    # Build the list of files with index
    file_info = [(f, i) for i, f in enumerate(files)]

    # Use multiprocessing pool
    Parallel(n_jobs=n_workers)(
        delayed(render_frame_parallel)(
            file_info[i],
            hold_frames, rot_frames, spiral_frames,
            thetas, phi, r_vals, output_dir
        ) for i in range(len(files))
    )

    # Combine to video
    dir_save = os.path.join(output_dir, "volume_rendering.mp4")
    frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    frames = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(dir_save, frames, fps=15)
    print(f"Video saved to: {dir_save}")


if __name__ == "__main__":
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")

    RUNS = os.getcwd()
    run_paths = glob.glob(os.path.join(RUNS, "out", "*prim.[0-9]*.phdf"))
    saveFile = run_paths[0].split("out")[0].split("ferhi/")[-1]
    if not os.path.exists(os.path.join('/u/ferhi/Figures/renders/',f"{saveFile}")): 
        os.makedirs(os.path.join('/u/ferhi/Figures/renders',f"{saveFile}"))
    saveDir = os.path.join('/u/ferhi/Figures/renders',f"{saveFile}")

    print("Output directory: ", saveDir)
    print("Outdir: ", saveDir)
    rot_and_spiral_zoom_in(run_paths, output_dir=saveDir, n_workers = N_procs //4 if N_procs > 4 else 1)