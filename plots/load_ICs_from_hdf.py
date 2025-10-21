import numpy as np
from read_hdf5 import read_hdf5 
import matplotlib.pyplot as plt
import os
from adjust_ics import *
from gen_strat import load_params, insert_sphere
from adioslib import *
import matplotlib

def ICs_from_file(file, filepath=None):

    data = read_hdf5(file, ['rho', 'prs', 'vel2'], filepath=filepath)
    full_rho = data['rho']
    full_prs = data['prs']  
    full_vel2 = data['vel2']


    return full_rho, full_prs, full_vel2


def reICs_file(file, filename_input='restrat.in', localDir='.', insert_sphere_radius = 0.0):
    
    full_rho, full_prs, full_vel2 = ICs_from_file(file, filepath=filename_input)

    params = load_params(filename_input)
    stratified_box = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbl1, mbl2, mbl3 = (int(params['reader'].get('parthenon/meshblock', f'nx{i}')) for i in range(1,4))
    code_length_cgs = float(params['reader'].get('units', 'code_length_cgs'))
    mbar_over_kb = stratified_box.mbar/ut.constants.kb

    mom = np.ones((nx1, nx2, nx3)) * full_rho * full_vel2   
    en = full_prs / (params['gamma'] - 1) + 0.5 * full_rho * full_vel2**2

    rho_cloud, mom_cloud, en_cloud = insert_sphere(full_rho, mom, en, r=params['r_cloud_inserted'] * code_length_cgs, 
                            T_cloud=params['T_cloud'], 
                            mbar_over_kb=mbar_over_kb, gamma=params['gamma'], T_base = params['T_base'],
                            x_range=(params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                            y_range=(params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                            z_range=(params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs),
                            inplace=False)

    fields = (rho_cloud, mom_cloud, en_cloud)
    mesh_shape = (nx1, nx2, nx3)
    mb_shape = (mbl1, mbl2, mbl3)

    reICs = gen_adios( mesh_shape, mb_shape, fields, 'reICs.bp', localDir)

    print(f"ICs shape: {reICs.shape}")
    try:
        plt.imshow(reICs[0, :, :, nx3 // 2], cmap='viridis', norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.savefig("reICs_slice.png")
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}. Maybe you have selected the wrong data type for ICs?")

    return reICs


if __name__ == "__main__":

    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'restrat.in')

    reICs_file('/viper/ptmp/ferhi/StratDisk/InfallingClouds/reference.phdf', filename_input, localDir)

