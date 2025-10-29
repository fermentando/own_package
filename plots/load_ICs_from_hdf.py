import numpy as np
from read_hdf5 import read_hdf5 
import matplotlib.pyplot as plt
import os
from adjust_ics import *
from gen_strat import load_params, insert_sphere
from adioslib import *
import matplotlib
import matplotlib.colors as colors


def ICs_from_file(file, filepath=None):

    data = read_hdf5(file, ['rho', 'prs', 'vel1', 'vel2', 'vel3'], filepath=filepath)
    full_rho = data['rho']
    full_prs = data['prs']  
    full_vel2 = data['vel2']
    full_vel1 = data['vel1']
    full_vel3 = data['vel3']


    return full_rho, full_prs, full_vel1, full_vel2, full_vel3


def reICs_file(file, filename_input='restrat.in', localDir='.', insert_sphere_radius = 0.0):
    
    full_rho, full_prs, full_vel1, full_vel2, full_vel3 = ICs_from_file(file, filepath=filename_input)
    
    params = load_params(filename_input)
    stratified_box = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    code_length_cgs = float(params['reader'].get('units', 'code_length_cgs'))
    mbar_over_kb = stratified_box.mbar/ut.constants.kb

    mom1 =  full_rho * full_vel1
    mom2 =  full_rho * full_vel2
    mom3 =  full_rho * full_vel3
    en = full_prs / (params['gamma'] - 1) + 0.5 * full_rho * (full_vel2*full_vel2 + full_vel1 * full_vel1 + full_vel3 * full_vel3)

    rho_cloud, mom1_cloud, mom2_cloud, mom3_cloud, en_cloud = insert_sphere(full_rho, mom1, mom2, mom3, en, r=params['r_cloud_inserted'] * code_length_cgs, 
                            T_cloud=params['T_cloud'], 
                            mbar_over_kb=mbar_over_kb, gamma=params['gamma'], T_base = params['T_base'],
                            x_range=(params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                            y_range=(params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                            z_range=(params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs),
                            inplace=False)

    fields = (rho_cloud,  mom1_cloud, mom2_cloud, mom3_cloud, en_cloud)
    mesh_shape = (nx1, nx2, nx3)
    mb_shape = (mbl1, mbl2, mbl3)

    reICs = gen_adios( mesh_shape, mb_shape, fields, 'reICs.bp', localDir)

    print(f"ICs shape: {reICs.shape}")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        # First subplot: x-component
        im0 = axes[0].imshow((mom1_cloud/full_rho)[:, :, nx3 // 2]/1e5, cmap='viridis')#, norm=colors.LogNorm())
        axes[0].set_title('Velocity X')
        plt.colorbar(im0, ax=axes[0])

        # Second subplot: y-component
        im1 = axes[1].imshow((mom2_cloud/full_rho)[:, :, nx3 // 2]/1e5, cmap='viridis')#, norm=colors.LogNorm())
        axes[1].set_title('Velocity Y')
        plt.colorbar(im1, ax=axes[1])

        # Third subplot: z-component
        im2 = axes[2].imshow((mom3_cloud/full_rho)[:, :, nx3 // 2]/1e5, cmap='viridis')#, norm=colors.LogNorm())
        axes[2].set_title('Velocity Z')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig("reICs_slice.png", dpi=300)
    except Exception as e:
        print(f"Error during plotting: {e}. Maybe you have selected the wrong data type for ICs?")

    return reICs


if __name__ == "__main__":

    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'restrat.in')

    reICs_file(os.path.join(localDir, 'reference.phdf'), filename_input, localDir)

