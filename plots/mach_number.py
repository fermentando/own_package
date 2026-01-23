import matplotlib.pyplot as plt
import numpy as np
from adjust_ics import *
from cooling import get_c_s
from stratified_box import StratifiedBox
from turbulent_box import TurbulentBox
import os
import utils as ut

SIM_DIR = os.getcwd()

data = np.loadtxt(os.path.join(SIM_DIR,"out/parthenon.out1.hst"))
try:
    run = StratifiedBox(os.path.join(SIM_DIR, "strat.in"), ".")
    t = data[:, 0] 
    output_mach = data[:, -1]
except:
    run = TurbulentBox(os.path.join(SIM_DIR, "turbulence.in"), ".")
    t = data[:, 0] 
    output_mach = data[:, -1]



try:
    T0 = float(run.reader.get('problem/stratified_box', 'T_base'))
except:
    rho0 = float(run.reader.get('problem/turbulence', 'rho0')) * run.code_mass_cgs / run.code_length_cgs**3
    p0 = float(run.reader.get('problem/turbulence', 'p0')) * run.code_mass_cgs / run.code_length_cgs / run.code_time_cgs**2
    T0 = p0 / (rho0 * ut.constants.kb / run.mbar)  # Initial temperature in K
    print("This is T0:", T0)
mach = float(run.reader.get('problem/turbulence', 'Mach_drive'))
k_peak = float(run.reader.get('problem/turbulence', 'kpeak'))
Lymin, Lymax = float(run.reader.get('parthenon/mesh', 'x2min')), float(run.reader.get('parthenon/mesh', 'x2max'))
Lxmin, Lxmax = float(run.reader.get('parthenon/mesh', 'x1min')), float(run.reader.get('parthenon/mesh', 'x1max'))
Lzmin, Lzmax = float(run.reader.get('parthenon/mesh', 'x3min')), float(run.reader.get('parthenon/mesh', 'x3max'))
L_drive = Lxmax - Lxmin
V_box = (Lxmax - Lxmin) * (Lymax - Lymin) * (Lzmax - Lzmin)
cs = get_c_s(T0)  # Sound speed in the medium
velocity_cgs = run.code_length_cgs / run.code_time_cgs
v_turb = cs * mach / velocity_cgs


#p0 = rho0 * ut.constants.kb * T0 / mbar  # Reference pressure
#p_floor = p0*0.00001

t_eddy = L_drive/v_turb 
print("This is last time / t_eddy:", t[-1]/t_eddy)

plt.plot(t/t_eddy, output_mach/V_box)
plt.xlabel("Time")  
plt.ylabel("Mach number")
#plt.ylim(0, 0.1)
print("saving figure mach_number.png")
plt.savefig("mach_number.png", dpi=200)