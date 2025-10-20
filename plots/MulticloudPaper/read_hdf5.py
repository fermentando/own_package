import h5py
import numpy as np
from joblib import Parallel, delayed
import numpy as np
#import h5pickle as h5py
import utils as ut
import os
import concurrent.futures
import os
import numpy as np
from joblib import Parallel, delayed

def read_hdf5(filename=None, fields=['rho'], n_jobs=1):
    if filename is None:
        filename = "/raven/ptmp/ferhi/ISM_slab/100kc/fv01e/out/parthenon.prim.00007.phdf"

    # Load code units outside the parallel block
    reader = ut.AthenaPKInputFileReader(os.path.abspath(os.path.join(filename, "../../ism.in")))
    code_units_length = float(reader.get('units', 'code_length_cgs'))
    code_units_time = float(reader.get('units', 'code_time_cgs'))
    code_units_mass = float(reader.get('units', 'code_mass_cgs'))
    code_units_vel = code_units_length / code_units_time
    code_units_rho = code_units_mass / code_units_length**3

    # Domain bounds
    xmin = float(reader.get("parthenon/mesh", "x1min"))
    xmax = float(reader.get("parthenon/mesh", "x1max"))
    ymin = float(reader.get("parthenon/mesh", "x2min"))
    ymax = float(reader.get("parthenon/mesh", "x2max"))
    zmin = float(reader.get("parthenon/mesh", "x3min"))
    zmax = float(reader.get("parthenon/mesh", "x3max"))

    with h5py.File(filename, "r") as f:
        prim = f["prim"]
        MeshBlockSize = f['Info'].attrs["MeshBlockSize"]
        RootGridSize = f['Info'].attrs["RootGridSize"]
        NumMeshBlocks = f['Info'].attrs["NumMeshBlocks"]
        LogicalLocations = f["LogicalLocations"][()]

        mbl3, mbl2, mbl1 = MeshBlockSize
        nx3, nx2, nx1 = RootGridSize

        nz, ny, nx = nx3, nx2, nx1
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        dz = (zmax - zmin) / nz

        # Build coordinate grid
        x = xmin + dx * (np.arange(nx) + 0.5)
        y = ymin + dy * (np.arange(ny) + 0.5)
        z = zmin + dz * (np.arange(nz) + 0.5)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")  # (nx, ny, nz)

        # Prepare output arrays
        data = {field: np.zeros((nz, ny, nx), dtype=np.float64) for field in fields}
        data['x1'] = X.transpose(2, 1, 0) * code_units_length
        data['x2'] = Y.transpose(2, 1, 0) * code_units_length
        data['x3'] = Z.transpose(2, 1, 0) * code_units_length

        def process_blocks(block_indices, filename, LogicalLocations, MeshBlockSize, code_units_rho, code_units_vel, fields):
            with h5py.File(filename, "r") as f:
                prim = f["prim"]
                results = []
                for i in block_indices:
                    lx, ly, lz = LogicalLocations[i]
                    mbl3, mbl2, mbl1 = MeshBlockSize
                    start1 = mbl1 * lx
                    start2 = mbl2 * ly
                    start3 = mbl3 * lz
                    end1 = start1 + mbl1
                    end2 = start2 + mbl2
                    end3 = start3 + mbl3

                    blocks = {}
                    for field in fields:
                        try:
                            if field == 'rho':
                                block = prim[i, 0, :, :, :] * code_units_rho
                            elif field == 'prs':
                                block = prim[i, 4, :, :, :] * code_units_rho * code_units_vel**2
                            elif field == 'vel1':
                                block = prim[i, 1, :, :, :] * code_units_vel
                            elif field == 'vel2':
                                block = prim[i, 2, :, :, :] * code_units_vel
                            elif field == 'vel3':
                                block = prim[i, 3, :, :, :] * code_units_vel
                            elif field == 'T':
                                rho_block = prim[i, 0, :, :, :] * code_units_rho
                                prs_block = prim[i, 4, :, :, :] * code_units_rho * code_units_vel**2
                                block = prs_block / rho_block / ut.constants.kb * ut.constants.mu * ut.constants.mh
                            else:
                                continue
                            blocks[field] = block
                        except Exception as e:
                            print(f"Error reading block {i}, field '{field}': {e}")
                            continue
                    results.append((start3, end3, start2, end2, start1, end1, blocks))
            return results


        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        block_indices = list(range(NumMeshBlocks))

        # Tune batch_size for your system, e.g., 5 or 10 blocks per batch
        batch_size = max(1, NumMeshBlocks // (n_jobs * 4))  # heuristic: 4 batches per worker
        block_chunks = list(chunks(block_indices, batch_size))

        results_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_blocks)(chunk, filename, LogicalLocations, MeshBlockSize, code_units_rho, code_units_vel, fields) for chunk in block_chunks
        )

        # Flatten results
        results = [item for sublist in results_chunks for item in sublist]

        for (start3, end3, start2, end2, start1, end1, blocks) in results:
            for field, block in blocks.items():
                data[field][start3:end3, start2:end2, start1:end1] = block



    return data
