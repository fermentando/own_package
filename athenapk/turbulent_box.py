"""
Turbulent box simulation setup and utilities.

Handles initialization and adjustment of turbulent simulations.
"""

import math
import numpy as np
import utils as ut
from cooling import initialize_cooling_constants


class TurbulentBox:
    """
    Setup and management for turbulent box simulations.
    
    Handles turbulent simulations with and without stratification.
    """
    
    def __init__(self, filename_input, dir):
        """
        Initialize the TurbulentBox simulation.
        
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
        self._set_t_corr()

    def _initialize_constants(self):
        """Initialize physical constants from input file."""
        self.gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        mu_H = 1.0
        mu = 1 / (He_mass_fraction * 3 / 4 + (1 - He_mass_fraction) * 2)
        self.mbar = mu * ut.constants.uam
        
        # Initialize cooling module with these constants
        initialize_cooling_constants(self.gamma, self.mbar)

    def _get_c_s(self, T):
        """
        Calculate sound speed.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin
            
        Returns
        -------
        float
            Sound speed in cm/s
        """
        return np.sqrt(self.gamma * ut.constants.kb * T / self.mbar)

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

    def _set_t_corr(self):
        """Set turbulence correlation time and output parameters (simple turbulence box)."""
        T0 = float(self.reader.get('problem/turbulence', 'T0'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        code_time_cgs = float(self.reader.get('units', 'code_time_cgs'))
        code_velocity_cgs = code_length_cgs / code_time_cgs
        L_box = Lymax - Lymin
        cs = self._get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach / code_velocity_cgs

        L_drive = L_box / k_peak
        t_eddy = L_drive / v_turb
        accel_rms = v_turb**2 / L_drive 

        tlim = 10 * t_eddy * code_time_cgs
        dt_hst = 0.0001 * t_eddy * code_time_cgs
        dt_hdf = 0.5 * t_eddy * code_time_cgs
        dt_rst = 0.5 * t_eddy * code_time_cgs
        
        self.reader.set_('problem/turbulence', 'corr_time', t_eddy)
        self.reader.set_('problem/turbulence', 'accel_rms', accel_rms)
        self.reader.set_('parthenon/time', 'tlim', tlim)
        self.reader.set_('parthenon/output1', 'dt', dt_hst)
        self.reader.set_('parthenon/output2', 'dt', dt_hdf)
        self.reader.set_('parthenon/output3', 'dt', dt_rst)
        self.reader.save()

        print(f"Driving correlation time set to {t_eddy:.3e} s")

    def _set_t_corr_stratified(self):
        """Set turbulence correlation time and output parameters (stratified turbulence box)."""
        T0 = float(self.reader.get('problem/stratified_box', 'T_base'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        L_box = Lymax - Lymin
        cs = self._get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach

        dfloor = 1e-24 * 0.01

        L_drive = L_box / k_peak
        t_eddy = L_drive / v_turb
        accel_rms = v_turb**2 / L_drive 

        tlim = 8 * t_eddy
        dt_hst = 0.01 * t_eddy
        dt_hdf = 0.1 * t_eddy
        dt_rst = 0.5 * t_eddy

        self.reader.set_('problem/turbulence', 'corr_time', t_eddy)
        self.reader.set_('hydro', 'dfloor', dfloor)
        self.reader.set_('problem/turbulence', 'accel_rms', accel_rms)
        self.reader.set_('parthenon/time', 'tlim', tlim)
        self.reader.set_('parthenon/output1', 'dt', dt_hst)
        self.reader.set_('parthenon/output2', 'dt', dt_hdf)
        self.reader.set_('parthenon/output3', 'dt', dt_rst)
        self.reader.save()
        print("this is dt_hdf: ", dt_hdf)
        print(f"Driving correlation time set to {t_eddy:.3e} s")

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
        for axis in axs:
            if axis == 2: fmin = -0.1; fmax = 0.9
            else: fmin = -0.5; fmax = 0.5
            xmin2, xmax2 = float(self.reader.get('parthenon/mesh', f'x{axis}min')), float(self.reader.get('parthenon/mesh', f'x{axis}max'))
            nx2_per_m = int(self.reader.get('parthenon/meshblock', f'nx{axis}'))
            meshblocks = int(self.reader.get('parthenon/mesh', f'nx{axis}')) / nx2_per_m
            if increase_factor > 1:
                enlarge_by = math.ceil(increase_factor * meshblocks)
            elif increase_factor <= 1:           
                enlarge_by = math.floor(increase_factor * meshblocks)
            cell_size = (xmax2 - xmin2) / int(self.reader.get('parthenon/mesh', f'nx{axis}'))
            self.reader.set_('parthenon/mesh', f'nx{axis}', nx2_per_m * enlarge_by)
            self.reader.set_('parthenon/mesh', f'x{axis}max', fmax*cell_size * nx2_per_m * enlarge_by)
            self.reader.set_('parthenon/mesh', f'x{axis}min', fmin*cell_size * nx2_per_m * enlarge_by)
            self.reader.save()
        self._enforce_cartesian_grid()
