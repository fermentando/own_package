"""
Turbulent box simulation setup and utilities.

Handles initialization and adjustment of turbulent simulations.
"""

import math
import numpy as np
import utils as ut
from cooling import initialize_cooling_constants, get_l_shatter, load_cooling_table, get_t_cool_n
import os


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
        self._load_cooling_table(dir)
        self._set_t_corr()

    def _initialize_constants(self):
        """Initialize physical constants from input file."""
        self.gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        self.m_H = 1.0 / (1 - He_mass_fraction)
        self.mu = 1 / (2 * (1 - He_mass_fraction) + He_mass_fraction * 3 / 4)
        self.mbar = self.mu * ut.constants.uam

        self.code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        self.code_time_cgs = float(self.reader.get('units', 'code_time_cgs'))
        self.code_mass_cgs = float(self.reader.get('units', 'code_mass_cgs'))
        
        # Initialize cooling module with these constants
        initialize_cooling_constants(self.gamma, self.mbar, self.mu, self.m_H)

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
        p0 = float(self.reader.get('problem/turbulence', 'p0')) * self.code_mass_cgs / self.code_length_cgs / self.code_time_cgs**2
        rho0 = float(self.reader.get('problem/turbulence', 'rho0')) * self.code_mass_cgs / self.code_length_cgs**3
        T0 = p0 / (rho0 * ut.constants.kb / self.mbar)  # Initial temperature in K
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        code_velocity_cgs = self.code_length_cgs / self.code_time_cgs
        L_box = Lymax - Lymin
        cs = self._get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach / code_velocity_cgs
        print("This is T0: ", T0)
        print("this is v_turb: ", v_turb)

        L_drive = L_box / k_peak
        t_eddy = L_drive / v_turb
        accel_rms = v_turb**2 / L_drive 

        tlim = 20 * t_eddy 
        dt_hst = 0.0001 * t_eddy
        dt_hdf = 0.5 * t_eddy
        dt_rst = 0.5 * t_eddy 
        t_injec = 3 * t_eddy
        
        self.reader.set_('problem/turbulence', 'inject_once_at_time',  t_injec)
        self.reader.set_('problem/turbulence', 'rescale_once_at_time',  t_injec)
        self.reader.set_('problem/turbulence', 'rescale_to_rms_Ms',  mach)
        self.reader.set_('cooling', 'start_time', t_injec)
        self.reader.set_('problem/turbulence', 'corr_time', t_eddy)
        self.reader.set_('problem/turbulence', 'accel_rms', accel_rms)
        self.reader.set_('parthenon/time', 'tlim', tlim)
        self.reader.set_('parthenon/output1', 'dt', dt_hst)
        self.reader.set_('parthenon/output2', 'dt', dt_hdf)
        self.reader.set_('parthenon/output3', 'dt', dt_rst)
        self.reader.save()

        print(f"Driving correlation time set to {t_eddy:.3e} Myr")

        self.t_eddy = t_eddy  # Store for external access

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


    def state_ICs(self):
        """Print the current state of the initial conditions."""
        Lambda_units = float(self.reader.get('cooling', 'lambda_units_cgs'))
        print(f"Cooling function units: {Lambda_units:.3e} erg cm^3 / s")
        p0 = float(self.reader.get('problem/turbulence', 'p0')) * self.code_mass_cgs / self.code_length_cgs / self.code_time_cgs**2
        rho0 = float(self.reader.get('problem/turbulence', 'rho0')) * self.code_mass_cgs / self.code_length_cgs**3
        T0 = p0 / (rho0 * ut.constants.kb / self.mbar)  # Initial temperature in K
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        dx = (float(self.reader.get('parthenon/mesh', 'x1max')) - float(self.reader.get('parthenon/mesh', 'x1min'))) / int(self.reader.get('parthenon/mesh', 'nx1'))
        rcl = float(self.reader.get('problem/turbulence', 'inject_blob_radius_0'))
        cs = self._get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach

        print("Current Initial Conditions:")
        print(f"  Density: {rho0:.3e} g/cm^3")
        print(f"  Pressure: {p0:.3e} dyne/cm^2")
        print(f"  Temperature: {T0:.3e} K")
        print(f"  Mean Molecular Weight: {self.mbar:.3e} g")
        print(f"  Sound Speed: {cs:.3e} cm/s")
        print(f"  Turbulent Velocity: {v_turb:.3e} cm/s (Mach {mach}) \n")

        print(f"  Density of the cloud: {rho0/self.mbar * 100:.3e} cm^-3")
        print(f" This is l_shatter: {get_l_shatter(p0, self.mbar)[0] / ut.constants.pc_to_cm:.3e} pc")
        print(f"  Cooling Time: {get_t_cool_n(T0 / 100, rho0 * 100 / self.mbar, self.mbar) * ut.constants.s_to_Myrs * Lambda_units:.3e} Myr")
        print(f"  t_cool,mix / t_eddy = {get_t_cool_n(T0 / 10, rho0 * 10 / self.mbar, self.mbar)*Lambda_units / (rcl * self.code_length_cgs * 10 / v_turb):.3e}")
        print(f"  Cloud radius: {rcl:.3e} pc")
        print(f"  Cloud / Lshatter: {rcl * ut.constants.pc_to_cm / get_l_shatter(p0)[0] / Lambda_units:.3e} ")
        print(f"  Cloud radius / cell resolution: {rcl /dx:.3e} ")


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

    def compute_restart_cooling_time(self):
        """Compute cooling time for restart conditions."""
        import re

        baseDir = self.filename.rsplit('/turbulence', 1)[0]
        print("Base dir: ", baseDir)
        slurm_file = os.path.join(baseDir, 'slurm')

        text = open(slurm_file).read()

        p0 = float(self.reader.get('problem/turbulence', 'p0')) * self.code_mass_cgs / self.code_length_cgs / self.code_time_cgs**2
        rho0 = float(self.reader.get('problem/turbulence', 'rho0')) * self.code_mass_cgs / self.code_length_cgs**3
        T0 = p0 / (rho0 * ut.constants.kb / self.mbar) 
        rcl = float(self.reader.get('problem/turbulence', 'inject_blob_radius_0'))
        chi = float(self.reader.get('problem/turbulence', 'inject_blob_chi_0'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))

        tlim = re.search(r"parthenon/time/tlim=([0-9.eE+-]+)", text)
        lambda_units = re.search(r"cooling/lambda_units_cgs=([0-9.eE+-]+)", text)
        t_cool_restart = get_t_cool_n(T0 / 10, rho0 * 10 / self.mbar) * float(lambda_units.group(1)) * ut.constants.s_to_Myrs
        print(f"Cooling time at restart conditions: {t_cool_restart:.3e} Myrs")
        print(f"tcool,mix / tcc at restart conditions: {get_t_cool_n(T0 / 10, rho0 * 10 / self.mbar)*float(lambda_units.group(1)) / (rcl * ut.constants.pc_to_cm * 10 /(mach * self._get_c_s(T0))):.3e}")
        lshatter = get_l_shatter(rho0/self.mbar * ut.constants.kb *1e6)[0] * float(lambda_units.group(1)) / ut.constants.pc_to_cm
        print(f"  Cloud / Lshatter: {rcl* ut.constants.pc_to_cm / get_l_shatter(rho0 / self.mbar * ut.constants.kb * T0)[0]/ float(lambda_units.group(1)):.3e} ")
        print(f"lshatter at restart conditions: {lshatter:.3e} pc")
        return t_cool_restart
