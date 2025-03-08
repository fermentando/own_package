import os
import yt 
import glob
import sys
import numpy as np
from utils import *
from adjustICs import *
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate

runDir = '/home/fernando/Runs/TestRuns/'
saveDir = '/home/fernando/Runs/'
RUNS = ['Tceiling_t0.001']#', 't10', 't0.1']#, 't0.01_mainrepo_v2']
COLOURS = ['crimson', 'black', 'slateblue', 'goldenrod', 'mediumseagreen']

def dTdt_isobaric(t, T):
    T_clipped = np.clip(T, 1e4, 1e6)
    dT = - (5/3 - 1) * 0.1 * 1e5 / T_clipped / ut.constants.kb * 10**f_interp(np.log10(T_clipped))
    return dT

def dTdt_isochoric(t, T):
    T_clipped = np.clip(T, 1e4, 1e6)
    dT = - (5/3 - 1) * 0.1 / ut.constants.kb * 10**f_interp(np.log10(T_clipped))
    return dT

for j, run in enumerate(RUNS):

    sim = SingleCloudCC(os.path.join(runDir+run, 'bin_cloud.in'), dir=runDir+run)
    code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
    if os.path.exists(os.path.join(run, "*hst")):
        data = np.loadtxt(os.path.join(runDir+run, 'parthenon.out1.hst'))
        norm_mass = np.log10(data[:, -1]/data[0, -1])
        timeseries = data[:, 0]/sim.tcc * code_time_cgs

        plt.plot(timeseries, norm_mass, color=COLOURS[j])

    files = np.sort(glob.glob(os.path.join(runDir+run, 'parthenon.prim.*.phdf')))
    rho_cloud = float(sim.reader.get('problem/cloud', 'rho_cloud_cgs'))
    rho_wind = float(sim.reader.get('problem/cloud', 'rho_wind_cgs'))
    T_wind = float(sim.reader.get('problem/cloud', 'T_wind_cgs'))
    He_mass_fraction = float(sim.reader.get('hydro', 'He_mass_fraction'))
    mu = 1 / (He_mass_fraction * 3 / 4 + (1 - He_mass_fraction) * 2)
    mbar = mu * ut.constants.uam

    n_mix = rho_cloud / mbar
    tcool= get_t_cool(n_mix, rho_wind/rho_cloud * 1e6)
    initial_mass = 0

    print("Runtime in tcool,mix: ", 10000*sim.tcc/tcool)
    
    table_tcool = os.path.join(saveDir, 'tcoolmix_curve.txt')
    if True:
        if os.path.exists(table_tcool):
            print("Reading table")
            tcool_curve = np.loadtxt(table_tcool)
            t_eval = tcool_curve[:, 0]; isobaric_T = tcool_curve[:,1]; isochoric_T = tcool_curve[:,2]
        
        else:
            print("Solving ODE")
            table_filename = os.path.join('/home/fernando/Runs/cooling_tables',sim.reader.get('cooling', 'table_filename').split('/')[-1])
            cooling_table = np.loadtxt(table_filename)
            f_interp = interpolate.interp1d(cooling_table[:, 0], cooling_table[:, 1], kind='linear', fill_value='extrapolate')

            # Define time axis
            t_span = (0, 3*tcool) 
            t_eval = np.linspace(t_span[0], t_span[1], 100)  

            # Solve the ODE
            rtol = 1e-6
            atol = 1e-6
            isochoric_solution = solve_ivp(dTdt_isochoric, t_span, [1e5] , t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol) 
            isobaric_solution = solve_ivp(dTdt_isobaric, t_span, [1e5] , t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol)

            isochoric_T = isochoric_solution.y[0]
            isobaric_T = isobaric_solution.y[0]

            np.savetxt(table_tcool, np.vstack((t_eval, isobaric_T, isochoric_T)).T)
    
        plt.plot(t_eval/tcool, isochoric_T, color=COLOURS[0], label='Isochoric cooling')
        plt.plot(t_eval/tcool, isobaric_T, color=COLOURS[1], label='Isobaric cooling')







    for filename in files[:20]:
        ds = yt.load(filename)
        temp = ds.all_data()[('gas', 'temperature')] 
        density = ds.all_data()[('gas', 'mass')]
        coldg = np.average(temp[temp <= 0.5e6], weights=density[temp<= 0.5e6])
        ts = ds.current_time


        plt.scatter(ts/tcool * code_time_cgs, coldg, color = COLOURS[0])




    
    

plt.title(RUNS[0])
plt.xlabel(r't [$t_{cool,mix}$]')
plt.ylabel(r'$ T [K]$')
plt.ylim( bottom = 1e3)
plt.yscale('log')
#plt.xlim(0, 20)
plt.legend()
plt.savefig(f'/home/fernando/Scripts/tcool_{RUNS[0]}.png')
plt.show()
