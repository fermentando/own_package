import numpy as np
import utils as ut
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib
import os
import errno
import sys
from scipy.optimize import minimize_scalar


localDir = os.getcwd()

def get_c_s(T):
    return np.sqrt(gamma * ut.constants.kb * T / mbar)

def get_t_cool(n, T):
    rho = n * mbar
    e = ut.constants.kb * T / (gamma - 1) / mbar
    log_lambda_cgs = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
    Lambda = 10**log_lambda_cgs 
    n_H = rho / (mu_H * ut.constants.mh)
    return rho * e / n_H**2 / Lambda

def get_l_shatter(P):
    def l_shatter_function(T):
        return (get_c_s(T) * get_t_cool(P / (ut.constants.kb * T), T))

    res = minimize_scalar(l_shatter_function, bounds=(1e4, 1e6), method='bounded')
    return res.fun , res.x

class SingleCloudCC:
    def __init__(self, filename_input, dir):
        self.filename = filename_input
        self.reader = ut.AthenaPKInputFileReader(filename_input)
        
        rel_path = self.reader.get('cooling', 'table_filename')
        self.code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        global gamma, mbar, mu_H
        gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        mu_H = 1.0

        self.R_cloud = float(self.reader.get('problem/wtopenrun', 'r0_cgs'))
        rho_cloud = float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
        rho_wind = float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs'))
        T_wind = float(self.reader.get('problem/wtopenrun', 'T_wind_cgs'))

        T_cloud = T_wind * rho_wind / rho_cloud
        mu = 1 / (He_mass_fraction * 3 / 4 + (1 - He_mass_fraction) * 2)
        mbar = mu * ut.constants.uam
        n_mix = np.sqrt(rho_wind * rho_cloud) / mbar

        try:
            v_wind = float(self.reader.get('problem/wtopenrun', 'v_wind_cgs'))
        except:
            Mach_wind = float(self.reader.get('problem/wtopenrun', 'Mach_wind'))
            v_wind = np.sqrt(gamma * ut.constants.kb * T_wind / mbar) * Mach_wind

        cooling_table = os.path.abspath(os.path.join(dir,rel_path))
        try:
            data = np.loadtxt(cooling_table)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cooling table not found: {cooling_table}")

        global cooling_table_logT_cgs, cooling_table_logLambda_cgs
        cooling_table_logT_cgs = data[:, 0]
        cooling_table_logLambda_cgs = data[:, 1]

        T_mix = np.sqrt(T_wind * T_cloud)
        self.tcoolmix = get_t_cool(n_mix, T_mix) 
        self.tcc = np.sqrt(rho_cloud / rho_wind) * self.R_cloud / v_wind 
        self.Rcrit_x_surv_ratio = self.tcoolmix * v_wind / np.sqrt(rho_cloud / rho_wind) 
        l_shatter = get_l_shatter(rho_wind / mbar * ut.constants.kb * T_wind)

        self.variables = [
        T_wind, T_cloud, T_wind/T_cloud, v_wind/1e5, T_mix, 
        self.Rcrit_x_surv_ratio*ut.constants.kpc_over_cm, 
        self.R_cloud*ut.constants.kpc_over_cm, 
        self.tcc * ut.constants.s_to_Myrs , 
        self.tcoolmix * ut.constants.s_to_Myrs,
        get_t_cool(rho_cloud/mbar, T_cloud) / self.tcc,
        self.R_cloud / l_shatter[0],
        self.Rcrit_x_surv_ratio / self.R_cloud
        ]
        

    def state_ICs(self):
        setup_confirmation = f"""
        >> Cloud properties << 
        T_wind = {self.variables[0]:.3e}
        T_cloud = {self.variables[1]:.3e}
        Overdensity = {self.variables[2]:.3g}
        V_wind (km/s) = {self.variables[3]:.3g}
        T_mix = {self.variables[4]:.3g}
        Cooling_table = {os.path.abspath(self.reader.get('cooling', 'table_filename'))}
        Critical radius (kpc) = {self.variables[5]:.3g}
        Current radius (kpc) = {self.variables[6]:.3g}
        t_cc in Myrs = {self.variables[7]:.3e}
        t_cool,mix in Myrs =  {self.variables[8]:.3e}
        T_cool,cold = {self.variables[9]:.3g} t_cc
        r0_cloud in cgs = {self.R_cloud:.3e} = {self.variables[10]:.3g} l_shatter
        t_coolmix / t_cc =  {self.variables[11]:.3g}
        """
        print(setup_confirmation)

    def reset_survival(self, ratio, rdx):

        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        xmin1 = float(self.reader.get('parthenon/mesh', 'x1min'))
        xmax1 = float(self.reader.get('parthenon/mesh', 'x1max'))

        adjusted_cloud_radius = self.Rcrit_x_surv_ratio / ratio
        axis_scaling = adjusted_cloud_radius / self.R_cloud 

        self.reader.set_('problem/wtopenrun', 'r0_cgs', adjusted_cloud_radius)

        for lims in [(f'x{i}min', f'x{i}max') for i in range(1,4)]:
            self.reader.change_aspect_xlim('parthenon/mesh', lims[0], axis_scaling)
            self.reader.change_aspect_xlim('parthenon/mesh', lims[1], axis_scaling)

   
        self.reader.save()

        print(f"""
        >>> Adjusting cloud radius to new survival criterion ....


        New radius (kpc) = {adjusted_cloud_radius * ut.constants.kpc_over_cm:.3e}
        t_coolmix / t_cc =  {self.Rcrit_x_surv_ratio / adjusted_cloud_radius :.3e}

        New input file succesfully saved in: {self.filename}
        """)

        

    def rclouddcell(self):
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        
        xmin2 = float(self.reader.get('parthenon/mesh', 'x2min'))
        xmax2 = float(self.reader.get('parthenon/mesh', 'x2max'))

        code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        
        r_over_d = self.R_cloud/self.code_length_cgs / ((xmax2 - xmin2)/nx2)
        L_over_r = (xmax2 - xmin2) / (self.R_cloud/self.code_length_cgs)
        print('R_cloud over cell size: {:.3f}'.format(r_over_d))
        print('Box y-length over cloud radius: {:.3f}'.format(L_over_r))

    def rescale_r_cell(self, ratio=8):
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        xmin1 = float(self.reader.get('parthenon/mesh', 'x1min'))
        xmax1 = float(self.reader.get('parthenon/mesh', 'x1max'))
        code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        
        cell_size = (xmax1-xmin1)/nx1
        axis_scaling = self.R_cloud / code_length_cgs / cell_size / ratio

        for lims in [(f'x{i}min', f'x{i}max') for i in range(1,4)]:
            self.reader.change_aspect_xlim('parthenon/mesh', lims[0], axis_scaling)
            self.reader.change_aspect_xlim('parthenon/mesh', lims[1], axis_scaling)

   
        self.reader.save()
        
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        xmin1 = float(self.reader.get('parthenon/mesh', 'x1min'))
        xmax1 = float(self.reader.get('parthenon/mesh', 'x1max'))
        
        res_R = self.R_cloud / code_length_cgs / cell_size
        print("R_cloud/cell_size adjusted to: ", res_R)
        

def plotLambda(table_filename):
    try:
        data = np.loadtxt(table_filename, skiprows=8)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cooling table not found: {table_filename}")

    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel('Log Temperature')
    plt.ylabel('Log Cooling Rate')
    plt.title('Cooling Curve')
    plt.show()


if __name__ == "__main__":


    sim = SingleCloudCC(os.path.join(localDir, 'ism.in'), dir=localDir)
    if str.lower(sys.argv[1]) == 'check':
        sim.state_ICs()
        sim.rclouddcell()
    elif str.lower(sys.argv[1]) == 'adjust':
        print(float(sys.argv[2]))
        sim.reset_survival(float(sys.argv[2]), 8)
    elif str.lower(sys.argv[1]) == 'resolution':
        sim.rescale_r_cell(ratio = float(sys.argv[2]) if len(sys.argv)==3 else 8)
    else: 
        raise ValueError("Invalid choice: pick amongst checking the current survival ratio, 'check', or adjusting to new ratio, 'adjust' followed by your new t_coolmix/t_cc value.")
        

    
