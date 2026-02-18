"""
Single cloud in a wind (CC) simulation setup and utilities.

Handles initialization and adjustment of single cloud simulations.
"""

import os
import numpy as np
import utils as ut
from cooling import initialize_cooling_constants, load_cooling_table, get_t_cool_n, get_l_shatter
from utils_physics import calculate_pressure, estimate_mach_from_v_wind


class SingleCloudCC:
    """
    Setup and management for single cloud in a wind simulations.
    
    This class handles cloud properties, cooling calculations, and mesh adjustments
    for cloud-wind interaction simulations.
    """
    
    def __init__(self, filename_input, dir):
        """
        Initialize the SingleCloudCC simulation.
        
        Parameters
        ----------
        filename_input : str
            Path to the input file
        dir : str
            Directory containing the input file
        """
        self.filename = filename_input
        self.dir = dir
        self.reader = ut.AthenaPKInputFileReader(filename_input)
        self._initialize_constants()
        self._load_simulation_parameters()
        self._load_cooling_table(dir)
        self._calculate_variables()

    def _initialize_constants(self):
        """Initialize physical constants from input file."""
        self.gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        self.mu_H = 1.0 / (1.0 - He_mass_fraction)
        self.mu = 1 / (2 * (1 - He_mass_fraction) + He_mass_fraction * 3 / 4)
        self.mbar = self.mu * ut.constants.uam
        print(f"Initialized constants: gamma={self.gamma}, mu_H={self.mu_H:.3f}, mu={self.mu:.3f}, mbar={self.mbar:.3e} g")
        
        # Initialize cooling module with these constants
        initialize_cooling_constants(self.gamma, self.mbar, self.mu, self.mu_H)

    def _load_simulation_parameters(self):
        """Load cloud and wind parameters from input file."""
        self.R_cloud = float(self.reader.get('problem/wtopenrun', 'r0_cgs'))
        self.rho_cloud = float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
        self.rho_wind = float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs'))
        self.T_wind = float(self.reader.get('problem/wtopenrun', 'T_wind_cgs'))
        self.T_cloud = self.T_wind * self.rho_wind / self.rho_cloud
        self.v_wind = self._get_wind_velocity()
        self.n_mix = np.sqrt(self.rho_wind * self.rho_cloud) / self.mbar

    def _get_wind_velocity(self):
        """
        Get wind velocity from input file.
        
        Attempts to read velocity directly, or calculates from Mach number.
        
        Returns
        -------
        float
            Wind velocity in cm/s
        """
        try:
            return float(self.reader.get('problem/wtopenrun', 'v_wind_cgs'))
        except:
            try:
                Mach_wind = float(self.reader.get('problem/wtopenrun', 'mach_wind'))
            except:
                Mach_wind = float(self.reader.get('problem/wtopenrun', 'Mach_wind'))
            return np.sqrt(self.gamma * ut.constants.kb * self.T_wind / self.mbar) * Mach_wind
    
    def _modify_shock_mach(self):
        """
        Estimate and update the shock Mach number in the input file.
        """
        pressure = calculate_pressure(self.T_wind, self.rho_wind, mbar=self.mbar)
        mach_est = estimate_mach_from_v_wind(self.v_wind, self.gamma, pressure, self.rho_wind)
        self.reader.set_('problem/wtopenrun', 'mach_shock', mach_est)
        self.reader.save()
        print('Mach shock: ', mach_est)

    def _load_cooling_table(self, dir):
        """
        Load cooling table from file path specified in input.
        
        Parameters
        ----------
        dir : str
            Directory containing the cooling table
        """
        rel_path = self.reader.get('cooling', 'table_filename')
        cooling_table_path = os.path.abspath(os.path.join(dir, rel_path))
        load_cooling_table(cooling_table_path)

    def _calculate_variables(self):
        """Calculate derived timescales and length scales."""
        T_mix = np.sqrt(self.T_cloud * float(self.reader.get('problem/wtopenrun', 'T_wind_cgs')))
        self.tcoolmix = get_t_cool_n( T_mix,self.n_mix, self.mbar)
        self.tcc = np.sqrt(float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs')) /
                           float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs'))) * \
                           self.R_cloud / self.v_wind
        self.Rcrit_x_surv_ratio = self.tcoolmix * self.v_wind / np.sqrt(
            float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs')) / float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs')))
        self.l_shatter, _ = get_l_shatter(float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs')) / self.mbar * \
                                       ut.constants.kb * float(self.reader.get('problem/wtopenrun', 'T_wind_cgs')))

    def state_ICs(self):
        """Print summary of initial conditions."""
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        R_internal_units = self.R_cloud / float(self.reader.get('units', 'code_length_cgs'))
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        out1 = float(self.reader.get('parthenon/output0', 'dt'))
        cell_size_y = (xmax2 - xmin2) / nx2
        cloud_to_cell_ratio = R_internal_units / cell_size_y
        cloud_to_cell_0_1_ratio = 1/10 * R_internal_units / cell_size_y
        length_y_to_cloud_ratio = (xmax2 - xmin2) / R_internal_units
        length_x_to_cloud_ratio = (xmax1 - xmin1) / R_internal_units
        fv = float(self.reader.get('problem/wtopenrun', 'fv'))
        depth = float(self.reader.get('problem/wtopenrun', 'depth'))
        tsh_out = out1 * self.tcc / depth / self.R_cloud * self.v_wind

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
        fv = {fv:.3g}
        depth = {depth:.3g} rcl
        output cadence (t_shock) = {tsh_out:.3g}
        Rcrit in pc = {self.Rcrit_x_surv_ratio * ut.constants.kpc_over_cm * 1e3:.3g}
        n0 = {self.rho_cloud/self.mbar:.3g}
        pressure = {self.rho_cloud/self.mbar * self.T_cloud:.3e} dyne/cm^2
        1/pressure = {1/(self.rho_cloud/self.mbar * self.T_cloud):.3e} cm^3/dyne
        T_cloud (^5/2) = {(self.T_cloud)**2.5:.3g}
        lamba = {1/self.n_mix/get_t_cool_n(self.T_cloud, self.n_mix, self.mbar) * ut.constants.kb * 1e4:.3g} erg cm^3/s
        r_crit = {self.tcoolmix * self.v_wind / 10 :.3g} cm
        mbar = {self.mbar:.3e} g
        """)

    def reset_survival(self, ratio, rdx=8):
        """
        Adjust cloud radius to meet a new survival criterion.
        
        Parameters
        ----------
        ratio : float
            New survival ratio (tcool_mix / tcc)
        rdx : int, optional
            Refinement level (default 8)
        """
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
        """Scale mesh dimensions by a factor."""
        for axis in ['x1', 'x2', 'x3']:
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}min', axis_scaling)
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}max', axis_scaling)
        self.reader.save()

    def _enforce_cartesian_grid(self):
        """Enforce a Cartesian grid by adjusting y-axis to match x-axis cell size."""
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
        """
        Enlarge domain dimensions.
        
        Parameters
        ----------
        increase_factor : float
            Factor by which to enlarge
        axs : list
            Axes to enlarge (1, 2, or 3)
        """
        import math
        for axis in axs:
            if axis == 2: fmin = -0.1; fmax = 0.9
            else: fmin = -0.5; fmax = 0.5
            xmin2, xmax2 = float(self.reader.get('parthenon/mesh', f'x{axis}min')), float(self.reader.get('parthenon/mesh', f'x{axis}max'))
            nx2_per_m = int(self.reader.get('parthenon/meshblock', f'nx{axis}'))
            meshblocks = int(self.reader.get('parthenon/mesh', f'nx{axis}')) / nx2_per_m
            if increase_factor > 1:
                enlarge_by = math.ceil(increase_factor*meshblocks)
            elif increase_factor <= 1:           
                enlarge_by = math.floor(increase_factor*meshblocks)
            cell_size = (xmax2 - xmin2) / int(self.reader.get('parthenon/mesh', f'nx{axis}'))
            self.reader.set_('parthenon/mesh', f'nx{axis}', nx2_per_m * enlarge_by)
            self.reader.set_('parthenon/mesh', f'x{axis}max', fmax*cell_size * nx2_per_m * enlarge_by)
            self.reader.set_('parthenon/mesh', f'x{axis}min', fmin*cell_size * nx2_per_m * enlarge_by)
            self.reader.save()
        self._enforce_cartesian_grid()
        
    def set_rin_res(self, resol_factor):
        """
        Set resolution for inner radius of cloud.
        
        Parameters
        ----------
        resol_factor : float
            Resolution factor
        """
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin3, xmax3 = float(self.reader.get('parthenon/mesh', 'x3min')), float(self.reader.get('parthenon/mesh', 'x3max'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        R_internal_units = self.R_cloud / float(self.reader.get('units', 'code_length_cgs'))
        cell_size = (xmax2 - xmin2)/nx2
        rescaled_size = 1/10 * R_internal_units / resol_factor / cell_size
        
        for i in [1, 2, 3]:
            self.reader.change_aspect_xlim('parthenon/mesh', f'x{i}min', rescaled_size)
            self.reader.change_aspect_xlim('parthenon/mesh', f'x{i}max', rescaled_size)
        self._enforce_cartesian_grid()
        
    def _return_ICs(self):
        """
        Read initial conditions from binary file.
        
        Returns
        -------
        tuple
            (ICs array, kval)
        """
        self._load_simulation_parameters()
        kval = self.tcoolmix / self.tcc
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        nx3 = int(self.reader.get('parthenon/mesh', 'nx3'))
        expected_shape = (nx1, nx2, nx3, 4)  
        dtype = np.float64  

        with open(os.path.join(self.dir, "ICs.bin"), "rb") as f:
            raw_data = f.read()

        # Convert bytes back to NumPy array
        ICs = np.frombuffer(raw_data, dtype=dtype).reshape(expected_shape)
        return ICs, kval
