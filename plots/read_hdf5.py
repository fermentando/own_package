import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils as ut
import os


def read_hdf5(filename=None, fields=['rho']):
    if filename is None:
        filename = (
        
            "/raven/ptmp/ferhi/ISM_slab/100kc/fv01e/out/parthenon.prim.00007.phdf"
        
        )
        
    reader = ut.AthenaPKInputFileReader(os.path.abspath(os.path.join(filename, "../../ism.in")))
    code_units_length = float(reader.get('units', 'code_length_cgs'))
    code_units_time = float(reader.get('units', 'code_time_cgs'))
    code_units_mass = float(reader.get('units', 'code_mass_cgs'))
    code_units_vel = code_units_length / code_units_time
    code_units_rho = code_units_mass / code_units_length**3
    
    with h5py.File(filename, "r") as f:

        prim = f["prim"][()]
        print(np.shape(prim))

        rho = prim[:, 0, :, :, :] * code_units_rho
        vel1 = prim[:, 1, :, :, :] * code_units_vel
        vel2 = prim[:, 2, :, :, :] * code_units_vel
        vel3 = prim[:, 3, :, :, :] * code_units_vel
        prs = prim[:, 4, :, :, :] * code_units_vel * code_units_vel * code_units_rho

        raw_hdf5 = {
            
            'rho': rho,
            'prs': prs,
            'vel1': vel1,
            'vel2': vel2,
            'vel3': vel3,
            
        }
        
        if 'T' in fields:
            
            raw_hdf5['T']=  prs / rho  / ut.constants.kb * ut.constants.mu * ut.constants.mh
            
        

        # x1 = f["x1v"][()]
        # x2 = f["x2v"][()]
        # x3 = f["x3v"][()]

        LogicalLocations = f["LogicalLocations"][()]

        # print(f"{np.shape(x1) = }")
        # print(f"{np.shape(x2) = }")
        # print(f"{np.shape(x3) = }\n")
        f_info = f['Info']
        MeshBlockSize = f_info.attrs["MeshBlockSize"]

        RootGridSize = f_info.attrs["RootGridSize"]

        NumMeshBlocks = f_info.attrs["NumMeshBlocks"]


    #rho_full = np.zeros(tuple(RootGridSize), dtype=float)
    # prs_full = np.zeros(tuple(RootGridSize), dtype=float)

    # x1_full = np.zeros(np.product(np.shape(x1)), dtype=float)
    # x2_full = np.zeros(np.product(np.shape(x2)), dtype=float)
    # x3_full = np.zeros(np.product(np.shape(x3)), dtype=float)
    data = {}
    
    for field in fields:
        f_placeholder = np.zeros(tuple(RootGridSize), dtype=float)
        data[field] = f_placeholder

        for i in range(NumMeshBlocks):
            start1 = MeshBlockSize[0] * LogicalLocations[i][0]
            end1 = start1 + MeshBlockSize[0]


            start2 = MeshBlockSize[1] * LogicalLocations[i][1]
            end2 = start2 + MeshBlockSize[1]

            start3 = MeshBlockSize[2] * LogicalLocations[i][2]
            end3 = start3 + MeshBlockSize[2]


            data[field][start3:end3, start2:end2, start1:end1] = raw_hdf5[field][i]

    #     prs_full[start3:end3, start2:end2, start1:end1] = prs[i]

    #     x1_full[start1:end1] = x1[i]
    #     x2_full[start2:end2] = x2[i]
    #     x3_full[start3:end3] = x3[i]

    # T = prs_full / rho_full

    return data

if __name__ == "__main__":
    rho = read_hdf5(fields=['T'])
    print(rho['T'])
    print(np.shape(rho['T']))