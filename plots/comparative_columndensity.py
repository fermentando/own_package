import os
import glob
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


    
if __name__ == "__main__":

    run_paths = [
        '/viper/ptmp2/ferhi/fvLism/01kc/fv02',
        '/viper/ptmp2/ferhi/fvLism/kc/fv01_shorter',
        '/viper/ptmp2/ferhi/fvLism/01kc/fv01_30r',
    ]
    
    mode = 'cold'
    N_procs = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    print(f"RUNS: {run_paths}")
    
    plt.style.use("custom_plot")

    for j, run in enumerate(run_paths):
        run_name = run  # Get the last part of the path
        #if "fv03_long" in run: continue
        print(run)
                
        sim = SingleCloudCC(os.path.join(run, 'ism.in'), dir=run)
        code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
        files = np.sort(glob.glob(os.path.join(run, 'out/parthenon.prim.*.phdf')))
        
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        dt = float(sim.reader.get('parthenon/output1', 'dt'))
        depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
        fv = float(sim.reader.get('problem/wtopenrun', 'fv'))
        t_scale =  dt / (0.1 + 0.1 * fv * depth)

        cache_path = os.path.join(run, f'{mode}_' + 'column_density.npz')
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            data = np.load(cache_path)
            t, col_dens, err_lower, err_upper = data['t'], data['col_dens'], data['err_lower'], data['err_upper']
            

            # Plot the central line
            plt.plot(np.array(t) * t_scale, 0.76 * col_dens / ut.constants.mh, label=mode, color='k')

            # Fill between error bars
            plt.fill_between(t * t_scale,
                            (col_dens - err_upper)*0.76 / ut.constants.mh,
                            (col_dens + err_upper)*0.76 / ut.constants.mh,
                            color='grey',
                            alpha=0.3)
            
            
        plt.ylabel(r'$ N_\mathrm{HII}$')
        plt.yscale('log')


        plt.xlabel(r't ')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'/u/ferhi/Figures/general_col_dens.png')
        plt.show()

