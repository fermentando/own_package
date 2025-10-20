import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from new_read import read_hdf5  # Replace with actual import
from utils import get_user_args, get_n_procs_and_user_args  # Replace with actual import
from adjust_ics import SingleCloudCC  # Replace with actual import
import sys




def detect_cold_box(temp, threshold=5e4, padding=5):
    cold_mask = temp < threshold
    coords = np.argwhere(cold_mask)
    if coords.size == 0:
        raise ValueError("No cold gas found.")
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return ymin, ymax
 


if __name__ == "__main__":
    

    plot_yt = True
    plot_hst = False
    mode = 'hot'  # or 'hot'
    
    user_args = get_user_args(sys.argv)
    
    if len(user_args) > 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}/Analysis"
        print('Saved to: ', saveFile)
        if not os.path.exists(os.path.join('/u/ferhi/Figures/',saveFile)): 
            os.makedirs(os.path.join('/u/ferhi/Figures/',saveFile))

    #run_paths = np.array([os.path.join(runDir, run) for run in RUNS])
    else:
        runDir = os.getcwd()
        run_paths = np.array([
            os.path.join(runDir, folder) 
            for folder in os.listdir(runDir) 
            if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder)) 
        ])
        parts = runDir.split('/')
        saveFile = f"{parts[-2]}/{parts[-1]}/Analysis"
        if not os.path.exists(os.path.join('/u/ferhi/Figures/',saveFile)):
            os.makedirs(os.path.join('/u/ferhi/Figures/',saveFile))



    
    N_procs = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    print(f"RUNS: {run_paths}")
    

    for j, run in enumerate(run_paths):
        run_name = run  
        print(run)
                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))

        results_file = os.path.join('/u/ferhi/Figures/', saveFile, "fv_fA.npz")

        # If file exists, load it
        if os.path.exists(results_file):
            existing = np.load(results_file, allow_pickle=True)
            snapshot_indices = list(existing['snapshot_indices'])
            filenames = list(existing['filenames'])
            fv_values = list(existing['fv_values'])
            fa_values = list(existing['fa_values'])
            start_idx = len(snapshot_indices)
            print(f"Resuming from snapshot {start_idx}")
        else:
            snapshot_indices = []
            filenames = []
            fv_values = []
            fa_values = []
            start_idx = 0

        for idx, file in enumerate(files[start_idx:], start=start_idx):
            print(f"Processing file {file} ({idx})")
            try:
                data = read_hdf5(file, fields=['T'], n_jobs = 6, chunk_size = 4)  
                temp = data['T']  
                ymin, ymax = detect_cold_box(temp)

                #fv calculation
                region = temp[:, ymin:ymax, :]
                fv = np.sum(region < 5e4) / region.size


                #f_A calculation
                cold_mask = temp[:, ymin:ymax, :] < 5e4  # shape: (z, y, x)
                cold_los = np.any(cold_mask, axis=(0, 1))  # shape: (x,)
                fa = np.sum(cold_los) / cold_los.size

            except Exception as e:
                print(f"Skipping {file}: {e}")
                y_extent = np.nan

            snapshot_indices.append(idx)
            filenames.append(os.path.basename(file))
            fv_values.append(fv)
            fa_values.append(fa)

            # Save progress to .npz
            np.savez_compressed(
                results_file,
                snapshot_indices=np.array(snapshot_indices),
                filenames=np.array(filenames),
                fv_values=np.array(fv_values),
                fa_values=np.array(fa_values)
            )
            
        data = np.load(results_file)
        plt.figure(figsize=(8, 5))
        plt.plot(data['snapshot_indices'], data['fv_values'],  color='k', label='fv')
        plt.plot(data['snapshot_indices'], data['fa_values'],  color='r', label='f_A')
        plt.yscale('log')
        plt.xscale('linear')
        plt.xlabel("Snapshot Index (time)")
        plt.ylabel(r'$fv/fA$')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join('/u/ferhi/Figures/', saveFile.split("Analysis")[0], "fv_fa.png")
        plt.savefig(fig_path)
        print("Saved plot to:", fig_path)