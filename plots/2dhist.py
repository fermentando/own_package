import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
import glob
import utils as ut

def plot_mass_weighted_hist(run, output_dir):
    index = int(run.split("/")[-1].split(".")[2])
    output_path = os.path.join(output_dir, f"mass_weighted_temperature_{index:03d}.png")
    if not os.path.exists(output_path):
        ds = yt.load(run)
        ad = ds.all_data()
            
        temp = np.array(ad[('gas', 'temperature')])
        vel = np.array(ad[('gas', 'velocity_magnitude')].to('km/s'))
        mass = np.array(ad[('gas', 'mass')])
        
        # Avoid log10 of zero or negative values
        temp[temp <= 0] = np.nan
        
        log_temp = np.log10(temp)
        log_vel = vel
        log_mass = mass
        
        log_temp = log_temp[log_temp > -np.inf]  # Filter out -inf values (log(0) or log(negative))
        log_vel = log_vel[log_vel > -np.inf]
    
        # Remove NaN values before histogram calculation
        valid_indices = ~np.isnan(log_temp) & ~np.isnan(log_vel)
        hist, x_edges, y_edges, im= plt.hist2d(log_vel[valid_indices], log_temp[valid_indices], 
                                                weights=mass[valid_indices], bins=100)


        # Plotting using imshow for better handling of LogNorm
        plt.figure(figsize=(8, 6))
        im = plt.imshow(hist.T, origin='lower', aspect='auto', 
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                        cmap='plasma', norm=LogNorm())
        plt.colorbar(label='Mass-weighted Count')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Log Temperature ')

        output_path = os.path.join(output_dir, f"2dhist_{index:03d}.png")
        plt.savefig(output_path)
        print('Saved to ', output_path)
        
        # Mass-weighted velocity histogram (1D)
        plt.figure(figsize=(8, 6))
        plt.hist(log_vel[valid_indices], bins=100, weights=log_mass[valid_indices], color='blue', alpha=0.7)
        plt.xlabel('Velocity [km/s]')
        plt.ylabel('Mass-weighted Count')
        plt.yscale('log')
        plt.title('Mass-weighted Velocity Histogram')
        
        output_path = os.path.join(output_dir, f"mass_weighted_velocity_{index:03d}.png")
        plt.savefig(output_path)
        print('Saved to ', output_path)
        plt.close()
        
        # Mass-weighted temperature histogram (1D)
        plt.figure(figsize=(8, 6))
        plt.hist(log_temp[valid_indices], bins=100, weights=log_mass[valid_indices], color='red', alpha=0.7)
        plt.xlabel('Log Temperature [K]')
        plt.ylabel('Mass-weighted Count')
        plt.yscale('log')
        plt.title('Mass-weighted Temperature Histogram')

        output_path = os.path.join(output_dir, f"mass_weighted_temperture_{index:03d}.png")
        plt.savefig(output_path)
        print('Saved to ', output_path)
        plt.close()



if __name__ == "__main__":
    RUNS = [os.getcwd()]
    print(RUNS)
    saveFile = ut.homeDir+f'/Figures/{RUNS[0].split('/')[-2]}/{RUNS[0].split('/')[-1]}'
    print(saveFile)
    
    N_procs = ut.get_n_procs()
    print(f"N_procs set to: {N_procs} processors.")
    
    #for run in RUNS:
    file_list = np.sort(glob.glob(os.path.join(RUNS[0], "out/*prim.[0-9]*.phdf")))
    ut.run_parallel(file_list, plot_mass_weighted_hist, N_procs, output_dir = saveFile)