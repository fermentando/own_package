import matplotlib.pyplot as plt
import numpy as np
from adjust_ics import *
from cooling import get_c_s
from stratified_box import StratifiedBox
import os


SIM_DIR = os.getcwd()

data = np.loadtxt(os.path.join(SIM_DIR,"out/parthenon.out1.hst"))
run = StratifiedBox(os.path.join(SIM_DIR, "strat.in"), ".")
L_drive = 100 * run.reader.get('units', 'code_length_cgs')
time_units = float(run.reader.get('units', 'code_time_cgs'))
t = data[:, 0] 
output_mach = data[:, -1]

T0 = float(run.reader.get('problem/stratified_box', 'T_base'))
#rho0 = float(run.reader.get('problem/turbulence', 'rho0'))
mach = float(run.reader.get('problem/turbulence', 'Mach_drive'))
k_peak = float(run.reader.get('problem/turbulence', 'kpeak'))
Lymin, Lymax = float(run.reader.get('parthenon/mesh', 'x2min')), float(run.reader.get('parthenon/mesh', 'x2max'))
Lxmin, Lxmax = float(run.reader.get('parthenon/mesh', 'x1min')), float(run.reader.get('parthenon/mesh', 'x1max'))
Lzmin, Lzmax = float(run.reader.get('parthenon/mesh', 'x3min')), float(run.reader.get('parthenon/mesh', 'x3max'))
L_box = Lxmax - Lxmin
V_box = (Lxmax - Lxmin) * (Lymax - Lymin) * (Lzmax - Lzmin)
cs = get_c_s(T0)  # Sound speed in the medium
velocity_cgs = run.code_length_cgs / run.code_times_cgs
v_turb = cs * mach / velocity_cgs


#p0 = rho0 * ut.constants.kb * T0 / mbar  # Reference pressure
#p_floor = p0*0.00001

L_drive = L_box/k_peak
t_eddy = L_drive/v_turb 
print("This is last time / t_eddy:", t[-1]/t_eddy)

plt.plot(t/t_eddy, output_mach/V_box)
plt.xlabel("Time")  
plt.ylabel("Mach number")
#plt.ylim(0, 0.1)
print("saving figure mach_number.png")
plt.savefig("mach_number.png", dpi=200)