import matplotlib.pyplot as plt
import numpy as np
import errno
import os


def plot_all(fn):
    plt.clf()
    for i, cf in enumerate(glob.glob("*/" + fn)):
        dat = np.loadtxt(cf)
        plt.loglog(dat[:,0], dat[:,1], label = cf)
    

plot_all("lambda.dat")
plt.xlabel("T (code units)")
plt.ylabel(r"$\Lambda$ (code units)")
plt.show()

for cf in ["townsend-fig1-0.1kev.dat", "townsend-fig1-10kev.dat", "townsend-fig1-3kev.dat",
           "townsend-fig1-0.3kev.dat", "townsend-fig1-1kev.dat"]: 
    plot_all(cf)


dat_old = np.loadtxt("/Users/maxbg/uni_aktuell/postdoc/hydro/athena/tests/cloudy_cooling_curve/powerlaw_fitting/WSS09/WSS09_n1_Z1.txt")
at_new = np.loadtxt("/Users/maxbg/uni_aktuell/postdoc/hydro/athena/tests/cloudy_cooling_curve/powerlaw_fitting/WSS09/WSS09_CIE_Z1.txt")
dat_GS07 = np.loadtxt("/Users/maxbg/uni_aktuell/postdoc/hydro/projects/mhd_cloud/analysis/tests/coolcurve_solar.dat")

dat2_SB15 = np.loadtxt("SB15/Metal_free_cooling.dat")
dat_SB15 = np.loadtxt("SB15/Total_metals_cooling.dat")

plt.loglog(dat_old[:,0], dat_old[:,2], label = "WSS09, z=0, Z=1, n=1")
plt.loglog(dat_new[:,0], dat_new[:,1], label = "WSS09, CIE, Z=1")
plt.loglog(dat_GS07[:,0], dat_GS07[:,4], label = "GS07, CIE, Z=1")
plt.loglog(dat_new[:,0], dat_SB15 + dat2_SB15, label = "SB15, Metals + non-Metals", linestyle = '--')

for table_filename in ['gnat-sternberg.cooling_1Z', 'schure.cooling_1.0Z']:
    cooling_table = os.path.abspath('/home/fernando/Runs/cooling_tables/'+table_filename)
    try:
        data = np.loadtxt(cooling_table, skiprows = 8)
    except: 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cooling_table) 

    plt.plot(data[:, 0], data[:,1], label=table_filename)
plt.xlabel('log10(T [K])')
plt.ylabel(r'log10($\Lambda$ [ergs $cm^{-3} s^{-1}$])')
plt.legend()
plt.savefig('cooling_tables.png')
plt.show()


