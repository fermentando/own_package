import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar
import utils as ut
import math

class SingleCloudCC:
    def __init__(self, filename_input, dir):
        self.filename = filename_input
        self.dir = dir
        self.reader = ut.AthenaPKInputFileReader(filename_input)
        self._initialize_constants()
        self._load_simulation_parameters()
        self._load_cooling_table(dir)
        self._calculate_variables()

    def _initialize_constants(self):
        global gamma, mbar, mu_H
        gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        mu_H = 1.0
        mu = 1 / (He_mass_fraction * 3 / 4 + (1 - He_mass_fraction) * 2)
        mbar = mu * ut.constants.uam

    def _load_simulation_parameters(self):
        self.R_cloud = float(self.reader.get('problem/wtopenrun', 'r0_cgs'))
        rho_cloud = float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
        rho_wind = float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs'))
        self.T_wind = float(self.reader.get('problem/wtopenrun', 'T_wind_cgs'))
        self.T_cloud = self.T_wind * rho_wind / rho_cloud
        self.v_wind = self._get_wind_velocity(rho_wind, self.T_wind)
        self.n_mix = np.sqrt(rho_wind * rho_cloud) / mbar

    def _get_wind_velocity(self, rho_wind, T_wind):
        try:
            return float(self.reader.get('problem/wtopenrun', 'v_wind_cgs'))
        except:
            Mach_wind = float(self.reader.get('problem/wtopenrun', 'Mach_wind'))
            return np.sqrt(gamma * ut.constants.kb * T_wind / mbar) * Mach_wind

    def _load_cooling_table(self, dir):
        rel_path = self.reader.get('cooling', 'table_filename')
        cooling_table_path = os.path.abspath(os.path.join(dir, rel_path))
        try:
            data = np.loadtxt(cooling_table_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cooling table not found: {cooling_table_path}")
        global cooling_table_logT_cgs, cooling_table_logLambda_cgs
        cooling_table_logT_cgs, cooling_table_logLambda_cgs = data[:, 0], data[:, 1]

    def _calculate_variables(self):
        T_mix = np.sqrt(self.T_cloud * float(self.reader.get('problem/wtopenrun', 'T_wind_cgs')))
        self.tcoolmix = get_t_cool(self.n_mix, T_mix)
        self.tcc = np.sqrt(float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs')) /
                           float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs'))) * \
                           self.R_cloud / self.v_wind
        self.Rcrit_x_surv_ratio = self.tcoolmix * self.v_wind / np.sqrt(
            float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs')) / float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs')))
        self.l_shatter = get_l_shatter(float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs')) / mbar * \
                                       ut.constants.kb * float(self.reader.get('problem/wtopenrun', 'T_wind_cgs')))

    def state_ICs(self):
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        R_internal_units = self.R_cloud / float(self.reader.get('units', 'code_length_cgs'))
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        cell_size_y = (xmax2 - xmin2) / nx2
        cloud_to_cell_ratio = R_internal_units / cell_size_y
        cloud_to_cell_0_1_ratio = 1/6 * R_internal_units / cell_size_y
        length_y_to_cloud_ratio = (xmax2 - xmin2) / R_internal_units
        length_x_to_cloud_ratio = (xmax1 - xmin1) / R_internal_units

        print(f"""
        >> Cloud properties <<
        T_wind = {self.T_wind:.3e}
        T_cloud = {self.T_cloud:.3e}
        V_wind (km/s) = {self.v_wind/1e5:.3g}
        Critical radius (kpc) = {self.Rcrit_x_surv_ratio * ut.constants.kpc_over_cm:.3g}
        Current radius (kpc) = {self.R_cloud * ut.constants.kpc_over_cm:.3g}
        R_cloud / cell_size = {cloud_to_cell_ratio:.3f}
        r_in / cell_size = {cloud_to_cell_0_1_ratio:.3f}
        Length_y / R_cloud = {length_y_to_cloud_ratio:.3f}
        Length_x / R_cloud = {length_x_to_cloud_ratio:.3f}
        t_coolmix / t_cc = {self.Rcrit_x_surv_ratio / self.R_cloud:.3g}
        """)

    def reset_survival(self, ratio, rdx=8):
        adjusted_radius = self.Rcrit_x_surv_ratio / ratio
        axis_scaling = adjusted_radius / self.R_cloud
        self.reader.set_('problem/wtopenrun', 'r0_cgs', adjusted_radius)
        self._scale_mesh(axis_scaling)
        self._enforce_cartesian_grid()
        print(f"""
        >>> Adjusting cloud radius to new survival criterion ...
        New radius (kpc) = {adjusted_radius * ut.constants.kpc_over_cm:.3e}
        t_coolmix / t_cc = {self.Rcrit_x_surv_ratio / adjusted_radius:.3e}
        Cartesina grid enforced.
        New input file successfully saved in: {self.filename}
        """)

    def _scale_mesh(self, axis_scaling):
        for axis in ['x1', 'x2', 'x3']:
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}min', axis_scaling)
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}max', axis_scaling)
        self.reader.save()

    def _enforce_cartesian_grid(self):
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        cell_size_x = (xmax1 - xmin1) / nx1
        cell_size_y = (xmax2 - xmin2) / nx2
        y_adjustment = cell_size_x * nx2 - (xmax2 - xmin2)
        self.reader.set_('parthenon/mesh', 'x2max', (xmax2 + abs(xmax2)/(xmax2 - xmin2)*y_adjustment))
        self.reader.set_('parthenon/mesh', 'x2min', (xmin2 - abs(xmin2)/(xmax2 - xmin2)*y_adjustment))
        self.reader.save()
        
    def enlarge_dim(self, increase_factor, axs):
        for axis in axs:
            if axis ==2: fmin = -0.1; fmax = 0.9
            else: fmin = -0.5; fmax = 0.5
            xmin2, xmax2 = float(self.reader.get('parthenon/mesh', f'x{axis}min')), float(self.reader.get('parthenon/mesh', f'x{axis}max'))
            nx2_per_m = int(self.reader.get('parthenon/meshblock', f'nx{axis}'))
            meshblocks = int(self.reader.get('parthenon/mesh', f'nx{axis}')) / nx2_per_m
            if increase_factor > 1:
                enlarge_by = math.ceil(increase_factor*meshblocks)
            elif increase_factor  <= 1:           
                enlarge_by = math.floor(increase_factor*meshblocks)
            cell_size = (xmax2 - xmin2) / int(self.reader.get('parthenon/mesh', f'nx{axis}'))
            self.reader.set_('parthenon/mesh', f'nx{axis}', nx2_per_m * enlarge_by)
            self.reader.set_('parthenon/mesh', f'x{axis}max', fmax*cell_size * nx2_per_m * enlarge_by)
            self.reader.set_('parthenon/mesh', f'x{axis}min', fmin*cell_size * nx2_per_m * enlarge_by)
            self.reader.save()
        self._enforce_cartesian_grid()
        
        
    def set_rin_res(self, resol_factor):
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin3, xmax3 = float(self.reader.get('parthenon/mesh', 'x3min')), float(self.reader.get('parthenon/mesh', 'x3max'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        R_internal_units = self.R_cloud / float(self.reader.get('units', 'code_length_cgs'))
        cell_size = (xmax2 - xmin2)/nx2
        rescaled_size = 1/6 * R_internal_units / resol_factor / cell_size
        
        for i in [1,2,3]:
            self.reader.change_aspect_xlim('parthenon/mesh', f'x{i}min', rescaled_size)
            self.reader.change_aspect_xlim('parthenon/mesh', f'x{i}max', rescaled_size)
        self._enforce_cartesian_grid()
        
    def _return_ICs(self):
        self._load_simulation_parameters()
        kval = self.tcoolmix/self.tcc
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        nx3 = int(self.reader.get('parthenon/mesh', 'nx3'))
        expected_shape = (nx1, nx2, nx3, 4)  
        dtype = np.float64  

        with open(os.path.join(self.dir,"ICs.bin"), "rb") as f:
            raw_data = f.read()

        # Convert bytes back to NumPy array
        ICs = np.frombuffer(raw_data, dtype=dtype).reshape(expected_shape)
        return ICs, kval

def get_c_s(T):
    return np.sqrt(gamma * ut.constants.kb * T / mbar)


def get_t_cool(n, T):
    rho = n * mbar
    e = ut.constants.kb * T / (gamma - 1) / mbar
    log_lambda = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
    Lambda = 10**log_lambda
    n_H = rho / (mu_H * ut.constants.mh)
    return rho * e / (n_H**2 * Lambda)


def get_l_shatter(P):
    def l_shatter_func(T):
        return get_c_s(T) * get_t_cool(P / (ut.constants.kb * T), T)

    res = minimize_scalar(l_shatter_func, bounds=(1e4, 1e6), method='bounded')
    return res.fun, res.x

if __name__ == "__main__":
    
    localDir = os.getcwd()
    sim = SingleCloudCC(os.path.join(localDir, 'ism.in'), dir=localDir)
    if str.lower(sys.argv[1]) == 'check':
        sim.state_ICs()
    elif str.lower(sys.argv[1]) == 'adjust':
        print(float(sys.argv[2]))
        sim.reset_survival(float(sys.argv[2]), 8)
    elif str.lower(sys.argv[1]) == 'enlarge_y':
        sim.enlarge_dim(increase_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 1,
                        axs=[2])
    elif str.lower(sys.argv[1]) == 'enlarge_x':
        sim.enlarge_dim(increase_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 1, 
                        axs = [1,3])
    elif str.lower(sys.argv[1]) == 'res':
        sim.set_rin_res(resol_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 8)
    else:
        raise ValueError("Invalid choice: pick amongst checking the current survival ratio, 'check', or adjusting to new ratio, 'adjust' followed by your new t_coolmix/t_cc value.")
