import numpy as np
import os
import logging
from scipy.spatial.distance import pdist
from utils import get_n_procs_and_user_args
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import glob


def _get_pdist(k, ipick = None):
    if ipick is None:
        return pdist(k)
    else:
        return pdist(k[ipick])


def _get_pdist_pbc(k, Lbox, ipick = None):
    """pdist with periodic bounday conditions.
    Uses still scipy pdist to be fast.

    L_box is float
    """
    if ipick is not None:
        return _get_pdist_pbc(k[ipick], Lbox)

    N, dim = k.shape
    dist_nd_sq = np.zeros(N * (N - 1) // 2)  # to match the result of pdist
    for d in range(dim):
        pos_1d = k[:, d][:, np.newaxis]  # shape (N, 1)
        dist_1d = pdist(pos_1d)  # shape (N * (N - 1) // 2, )
        dist_1d[dist_1d > 0.5 * Lbox] -= Lbox
        dist_nd_sq += dist_1d**2 

    return np.sqrt(dist_nd_sq)

def velocity_structure_function(run_path, outdir,
                                cut_regions=[('all', None)], maxpoints=2e4, nbins=100,
                                percentiles=[16, 50, 84], plot=True):
    """
    Computes the velocity structure function for a given run directory.

    Parameters:
    run_path     -- Path to the simulation output folder
    outdir       -- Directory to store output npz and plots
    cut_regions  -- List of (name, cut_string) tuples for region selection
    maxpoints    -- Maximum number of points to use
    nbins        -- Number of distance bins
    percentiles  -- Percentiles to compute
    plot         -- Whether to generate a plot
    """
    import yt

    # Load latest dataset
    latest_file = sorted(glob.glob(os.path.join(run_path, 'out/parthenon.prim.*.phdf')))[-1]
    ds = yt.load(latest_file)
    ad = ds.all_data()

    #run_name = os.path.basename(run_path.strip('/'))
    #outdir = os.path.join(outdir, run_name)


    if plot:
        plotoutdir = os.path.join(outdir, "plots/")
        plt.clf()

    plotted = False
    for cname, cut_string in cut_regions:
        ofn = os.path.join(outdir, f"{str(ds)}_{cname}.npz")
        if os.path.isfile(ofn):
            logging.info("%s exists already. Skipping VSF computation.", ofn)
            continue
        logging.info("Computing VSF for %s [%s] --> %s", str(ds), cname, ofn)

        cad = ad if not cut_string else ad.cut_region(cut_string)

        pos = np.array([cad[ii].value for ii in ['x', 'y', 'z']]).T
        npoints = pos.shape[0]
        if npoints == 0:
            logging.warning("No points!")
            continue

        ipoints = np.random.permutation(npoints)[:int(maxpoints)] if npoints > maxpoints else None

        ds.index.clear_all_data()
        dists = _get_pdist_pbc(pos, ipoints)
        del pos

        vels = np.array([cad['velocity_' + ii].value for ii in ['x', 'y', 'z']])
        if ipoints is not None:
            vels = vels[:, ipoints]
        vels = vels.T
        del ipoints
        ds.index.clear_all_data()
        veldiffs = _get_pdist_pbc(vels)
        del vels

        dist_bins = np.logspace(np.log10(dists.min()), np.log10(dists.max()), nbins)
        ibins = np.digitize(dists, dist_bins)

        nums, qs, means = [], [], []
        for i in range(nbins):
            m = ibins == i
            nums.append(np.sum(m))
            means.append(np.mean(veldiffs[m]) if np.sum(m) > 0 else np.nan)
            qs.append(np.percentile(veldiffs[m], percentiles) if np.sum(m) > 0 else [np.nan]*len(percentiles))

        #np.savez(ofn, distance_bins=dist_bins, velocity_means=means, velocity_percentiles=qs, number=nums)

        if plot:
            plt.plot(dist_bins, means, label=cname)
            plt.fill_between(dist_bins, np.array(qs)[:, 0], np.array(qs)[:, -1], alpha=0.2)
            plotted = True

    if plot and plotted:
        plt.legend(loc='best')
        plt.loglog()
        plt.xlabel("d")
        plt.ylabel("<|v|>")
        ofn_plot = os.path.join(plotoutdir, f"{str(ds)}.png")
        plt.savefig(ofn_plot, bbox_inches='tight')


if __name__ == "__main__":
    
    
    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    
    if len(user_args) == 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
        print('Saved to: ', saveFile)
        if not os.path.exists(os.path.join('/u/ferhi/Figures/velocity_structure_function/',f"{parts[-3]}/{parts[-2]}")): 
            os.makedirs(os.path.join('/u/ferhi/Figures/velocity_structure_function/',f"{parts[-3]}/{parts[-2]}"))

    #run_paths = np.array([os.path.join(runDir, run) for run in RUNS])
    else:
        runDir = os.getcwd()
        run_paths = np.array([
            os.path.join(runDir, folder) 
            for folder in os.listdir(runDir) 
            if os.path.isdir(os.path.join(runDir, folder)) and 'ism.in' in os.listdir(os.path.join(runDir, folder)) 
        ])
        parts = runDir.split('/')
        saveFile = f"{parts[-2]}/{parts[-1]}"
        if not os.path.exists(os.path.join('/u/ferhi/Figures/velocity_structure_function/',parts[-2])): 
            os.makedirs(os.path.join('/u/ferhi/Figures/velocity_structure_function/',parts[-2]))

    print(run_paths)
    Parallel(n_jobs=N_procs)(
    delayed(velocity_structure_function)(run, saveFile) for run in run_paths
)