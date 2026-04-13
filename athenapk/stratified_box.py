"""
Stratified box simulation setup and utilities.

Handles initialization and adjustment of stratified disk simulations.
"""

import os
import math
import numpy as np
import utils as ut
from cooling import initialize_cooling_constants, load_cooling_table, get_t_cool_n, get_t_cool_cgs, get_l_shatter
from utils_physics import calculate_pressure


class StratifiedBox:
    """
    Setup and management for stratified disk simulations.
    
    Handles cloud insertion in a stratified ambient medium with gravity.
    """
    
    def __init__(self, filename_input, dir):
        """
        Initialize the StratifiedBox simulation.
        
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
        type_box = str(self.reader.get('job', 'problem_id'))
        self._initialize_constants()
        self._load_simulation_parameters()
        self._load_cooling_table(dir)
        self._cloud_conditions()
        self._enforce_cartesian_grid()
        self._timescales()
        if 'simple' not in type_box: 
            self._set_t_corr()

    def _initialize_constants(self):
        """Initialize physical constants from input file."""
        self.gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        self.mu_H = 1.0 / (1.0 - He_mass_fraction)
        self.mu = 1 / (2 * (1 - He_mass_fraction) + He_mass_fraction * 3 / 4)
        self.mbar = self.mu * ut.constants.uam

        # Initialize cooling module with these constants
        initialize_cooling_constants(self.gamma, self.mbar, self.mu, self.mu_H)

    def _load_simulation_parameters(self):
        """Load stratified box parameters from input file."""
        self.surface_density = float(self.reader.get('problem/stratified_box', 'surface_density'))
        self.T_base = float(self.reader.get('problem/stratified_box', 'T_base'))
        self.a_over_H = float(self.reader.get('problem/stratified_box', 'a_over_H'))
        self.r_cloud_inserted = float(self.reader.get('problem/stratified_box', 'r_cloud_inserted'))
        self.T_cloud = float(self.reader.get('problem/stratified_box', 'T_cloud'))
        self.chi = self.T_base / self.T_cloud
        self.mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))

        self.code_mass_cgs = float(self.reader.get('units', 'code_mass_cgs'))
        self.code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        self.code_time_cgs = float(self.reader.get('units', 'code_time_cgs'))

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

    def _cloud_conditions(self):
        """Calculate cloud and ambient density profiles."""
        mbar_over_kb = self.mbar / ut.constants.kb 

        self.g = 2 * np.pi * ut.constants.G * self.surface_density * self.code_mass_cgs / (self.code_length_cgs)**2
        c_s = np.sqrt(self.T_base / mbar_over_kb)
        self.H = c_s**2 / self.g
        self.rho_base = (self.surface_density * self.code_mass_cgs / (self.code_length_cgs)**2) / (2 * self.a_over_H * self.H)
        
        wmax = float(self.reader.get('problem/turbulence', 'window_xmax'))
        self.y_centre = float(self.reader.get('parthenon/mesh', 'x2min')) + 0.7 * ( float(self.reader.get('parthenon/mesh', 'x2max')) - float(self.reader.get('parthenon/mesh', 'x2min')) )
        self.env_rho = self.rho_base * np.exp(-self.a_over_H * (np.sqrt(1 + (self.y_centre * self.code_length_cgs / (self.a_over_H * self.H))**2) - 1))
        self.cloud_rho = self.chi * self.env_rho

        self.reader.set_('problem/stratified_box', 'loc_cloud_inserted', [0, self.y_centre, 0])
        self.reader.save()

    def _timescales(self):
        """Calculate important timescales for cloud growth and evolution."""
        Myr = 3.154e13            # seconds
        pc = 3.086e18  
        g_fid = 1e-8               # cm/s^2
        r_100 = 100.0              # reference radius
        c_s_150 = 150e5 

        self.t_cool_cl = get_t_cool_cgs(self.cloud_rho, self.T_cloud, self.mbar)
        self.t_grow_subsonic = (
            40 * Myr
            * (self.g / g_fid)**(-3/8)
            * (self.chi / 100)**(5/8)
            * (self.r_cloud_inserted / r_100)**(15/32)
            * (self.t_cool_cl / (0.03 * Myr))**(5/32)
        )

        self.t_grow_supersonic = (
            35 * Myr
            * (self._get_c_s(self.T_base) / c_s_150)**(-3/5)
            * (self.chi / 100)
            * (self.r_cloud_inserted / r_100)**(3/4)
            * (self.t_cool_cl / (0.03 * Myr))**(1/4)
        )

        self.t_ff = self._get_c_s(self.T_base) / self.g

        self.r_sonic = (
            150 * pc
            * (self.t_cool_cl / (0.03 * Myr))**(-1/3)
            * (self.g / g_fid)**(-4/3)
            * (self.chi / 100)**(-4/3)
            * (self._get_c_s(self.T_base) / c_s_150)**(32/15)
        )

        self.r_ss = (
            100 * pc
            * (self.t_cool_cl / (0.03 * Myr))**(-1)
            * (self.g / g_fid)**(-2)
            * (self.chi / 100)**(-2)
            * (self._get_c_s(self.T_base) / c_s_150)**(12/5)
        )

        self.r_ratio = (
            1.5
            * (self.t_cool_cl / (0.03 * Myr))**(2/3)
            * (self.g / g_fid)**(2/3)
            * (self.chi / 100)**(2/3)
            * (self._get_c_s(self.T_base) / c_s_150)**(-4/15)
        )

        self.t_surv_sub = (5e-3 * Myr / self.t_cool_cl) * (self.r_cloud_inserted / 100 ) **(1/5) * (self.g / g_fid) **(-4/5) * (self.chi / 100 ) **(-12/5)

        self.v_drag = np.sqrt(2 * self.chi * self.r_cloud_inserted * ut.constants.pc_to_cm * self.g / 0.47)
        
        self.v_grow_sub = self.g * self.t_grow_subsonic

        #self.t_cc = self.r_cloud_inserted * ut.constants.pc_to_cm * self.chi**0.5 / self._get_c_s(self.T_base)

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

    def stateICs(self):
        """Print summary of initial conditions."""
        nx2 = float(self.reader.get('parthenon/mesh', 'nx2'))
        x2min, x2max = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        dcell = (x2max - x2min) / nx2
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lxmin, Lxmax = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        Ldrive = (Lxmax - Lxmin)  * self.code_length_cgs
        Lx_over_rcl = (Lxmax - Lxmin) * self.code_length_cgs / (self.r_cloud_inserted * self.code_length_cgs)
        vs = self._get_c_s(self.T_base) * mach 
        t_eddy = Ldrive / vs 
        self.t_r_eddy = self.r_cloud_inserted * self.code_length_cgs / vs/ (Lx_over_rcl)**0.333333 * (self.chi)**0.5
        Lambda_units = float(self.reader.get('cooling', 'lambda_units_cgs'))
        tgrow = self.chi * np.sqrt(get_t_cool_cgs(self.cloud_rho, self.T_cloud, self.mbar)*Lambda_units * self.r_cloud_inserted * self.code_length_cgs / (vs))
        
        print(f"""
        >> Strat disk properties <<

        H = {self.H/ut.constants.pc_to_cm:.3e} pc
        g = {self.g:.3e} cgs
        r_cl / d_cell = {self.r_cloud_inserted /dcell:.3e} 
        L_drive / H = {Ldrive / self.H:.3e}
        y / H = {self.y_centre*ut.constants.pc_to_cm / self.H:.3e}
        Fr = {self.H / Ldrive * mach}
        
        R_cl = {self.r_cloud_inserted:.3e} pc
        n_cl = {self.cloud_rho/self.mbar:.3e} cm^-3
        rho_cl [code units]= {self.cloud_rho / (self.code_mass_cgs / self.code_length_cgs**3):.3e} code units
        n_0 = {self.rho_base/self.mbar:.3e} cm^-3
        chi = {self.chi}
        Ly / rcl = {(x2max - x2min) * self.code_length_cgs / (self.r_cloud_inserted * self.code_length_cgs):.3e}

        vs = {vs/1e5:.3e} km/s
        t_cool,mix = {get_t_cool_cgs(np.sqrt(self.env_rho * self.cloud_rho), np.sqrt(self.T_base * self.T_cloud), self.mbar)*Lambda_units*ut.constants.s_to_Myrs:.3e} Myr
        t_cool = {get_t_cool_cgs(self.cloud_rho, self.T_cloud, self.mbar)*Lambda_units*ut.constants.s_to_Myrs:.3e} Myr
        t_r_eddy = {self.t_r_eddy*ut.constants.s_to_Myrs:.3e} Myr
        t_cool,mix / t_eddy = {get_t_cool_cgs(np.sqrt(self.env_rho * self.cloud_rho), np.sqrt(self.T_base * self.T_cloud), self.mbar)*Lambda_units / self.t_r_eddy:.3e}
        t_grow = {self.t_grow_subsonic*ut.constants.s_to_Myrs:.3e} Myr
        Cloud / Lshatter: {self.r_cloud_inserted * self.code_length_cgs / get_l_shatter(self.env_rho/ self.mbar * ut.constants.kb * self.T_base)[0] /Lambda_units:.3e}

        Density: {self.env_rho:.3e} g/cm^3
        Temperature: {self.T_base:.3e} K
        Pressure: {self.env_rho * ut.constants.kb * self.T_base / self.mbar:.3e} cgs
        Mbar: {self.mbar:.3e} g
        Env_rho: {self.env_rho/self.mbar:.3e} cm^3

        tgrow = {tgrow * ut.constants.s_to_Myrs:.3e} Myr
        teddy = { t_eddy * ut.constants.s_to_Myrs:.3e} Myr
        tff = {np.sqrt((Lymax - Lymin) * self.code_length_cgs / self.g ) * ut.constants.s_to_Myrs:.3e} Myr
        tff,growth = {(Lymax - Lymin)/2 * self.code_length_cgs / (self.g * tgrow) * ut.constants.s_to_Myrs:.3e} Myr
        tff,drag = {(Lymax - Lymin)/2 * self.code_length_cgs / np.sqrt(self.g * self.chi * self.r_cloud_inserted * self.code_length_cgs * 2) * ut.constants.s_to_Myrs:.3e} Myr
        gtgrow = {self.g * tgrow / 1e5} kms^-1

        """)

    def _scale_mesh(self, axis_scaling):
        """Scale mesh dimensions by a factor."""
        for axis in ['x1', 'x2', 'x3']:
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}min', axis_scaling)
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}max', axis_scaling)
        self.reader.save()

    def _enforce_cartesian_grid(self):
        """Enforce a Cartesian grid by adjusting x-axis to match y-axis cell size."""
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        cell_size_y = (xmax2 - xmin2) / nx2
        y_adjustment = cell_size_y * nx1 - (xmax1 - xmin1)
        for axis in ['1', '3']:
            self.reader.set_('parthenon/mesh', f'x{axis}max', (xmax1 + abs(xmax1)/(xmax1 - xmin1)*y_adjustment))
            self.reader.set_('parthenon/mesh', f'x{axis}min', (xmin1 - abs(xmin1)/(xmax1 - xmin1)*y_adjustment))
            self.reader.save()

    def _enforce_cartesian_grid_on_y(self):
        """Enforce a Cartesian grid by adjusting y-axis to match x-axis cell size."""
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        # compute cell size from x1 and adjust x2 to match it
        cell_size_x = (xmax1 - xmin1) / nx1
        x_adjustment = cell_size_x * nx2 - (xmax2 - xmin2)
        delta2 = (xmax2 - xmin2)
        if delta2 == 0:
            # avoid division by zero; nothing to adjust
            return
        self.reader.set_('parthenon/mesh', 'x2max', (xmax2 + abs(xmax2) / delta2 * x_adjustment))
        self.reader.set_('parthenon/mesh', 'x2min', (xmin2 - abs(xmin2) / delta2 * x_adjustment))
        self.reader.save()
    
    def _set_t_corr(self):
        """Set turbulence correlation time and other time-dependent parameters."""
        T0 = float(self.reader.get('problem/stratified_box', 'T_base'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lxmin, Lxmax = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        code_time_cgs = float(self.reader.get('units', 'code_time_cgs'))
        code_velocity_cgs = code_length_cgs / code_time_cgs
        L_box = Lxmax - Lxmin
        cs = self._get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach / code_velocity_cgs
        self.t_cc = self.chi**0.5 * (self.r_cloud_inserted * ut.constants.pc_to_cm) / (cs * mach) 

        L_drive = L_box / k_peak
        self.t_eddy = L_drive / v_turb
        accel_rms = v_turb**2 / L_drive 
        cs_h = self._get_c_s(T0) * (self.chi/100)**0.5 
        tff = cs_h / self.g / code_time_cgs
        print("this is tff:, " ,tff)

        t_injec = 4 * self.t_eddy
        self.t_inject = t_injec
        tlim =  t_injec + 1000*self.t_eddy
        dt_hst = 0.001 * self.t_eddy
        dt_hdf = 0.1 * self.t_eddy
        dt_rst = 1 * self.t_eddy
        
        self.reader.set_('problem/stratified_box', 'rescale_code_time_to_tff', 'false')
        #self.reader.set_('problem/stratified_box', 'rescale_once_at_time',  t_injec)
        #self.reader.set_('problem/stratified_box', 'rescale_to_rms_Ms', mach)
        self.reader.set_('problem/stratified_box', 'inject_once_at_time',  t_injec)
        self.reader.set_('cooling', 'start_time', t_injec)
        self.reader.set_('problem/turbulence', 'corr_time', self.t_eddy)
        self.reader.set_('problem/turbulence', 'accel_rms', accel_rms)
        self.reader.set_('parthenon/time', 'tlim', tlim)
        self.reader.set_('parthenon/output1', 'dt', dt_hst)
        self.reader.set_('parthenon/output2', 'dt', dt_hdf)
        self.reader.set_('parthenon/output3', 'dt', dt_rst)

        loc = [0, self.y_centre, 0]
        self.reader.set_('problem/stratified_box', 'loc_cloud_inserted', loc)
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

    def rescale_to_H(self):
        """Rescale domain to match scale height H."""
        box_height = 6 * self.H / self.code_length_cgs
        fpos = 0.9
        fneg = -0.1
        nx2 = int(self.reader.get('parthenon/meshblock', 'nx2'))
        dim_cell = box_height / nx2
        for axis in ['2']:
            self.reader.set_('parthenon/mesh', f'x{axis}max', fpos * box_height)
            self.reader.set_('parthenon/mesh', f'x{axis}min', fneg * box_height)
            self.reader.save()
        for axis in ['1', '3']:
            nx = int(self.reader.get('parthenon/meshblock', f'nx{axis}'))
            self.reader.set_('parthenon/mesh', f'x{axis}min', -0.5 * nx * dim_cell)
            self.reader.set_('parthenon/mesh', f'x{axis}max', 0.5 * nx * dim_cell)
            self.reader.save()
        self._enforce_cartesian_grid()
        print("Rescaled succesfully.")
        
    def set_rin_res(self, resol_factor):
        """
        Set resolution for cloud inner radius.
        
        Parameters
        ----------
        resol_factor : float
            Resolution factor
        """
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin3, xmax3 = float(self.reader.get('parthenon/mesh', 'x3min')), float(self.reader.get('parthenon/mesh', 'x3max'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        R_internal_units = self.r_cloud_inserted
        cell_size = (xmax2 - xmin2) / nx2
        rescaled_size = R_internal_units / resol_factor / cell_size
        
        for i in [1, 2, 3]:
            self.reader.change_aspect_xlim('parthenon/mesh', f'x{i}min', rescaled_size)
            self.reader.change_aspect_xlim('parthenon/mesh', f'x{i}max', rescaled_size)
        self._enforce_cartesian_grid()

    def _get_t_cool_min(self, rho, T, mbar_val, Tmin=1e4, Tmax=1e6):
        """
        Find the minimum cooling time (helper method).
        
        Parameters
        ----------
        rho : float
            Density in g/cm^3
        T : float
            Temperature in Kelvin
        mbar_val : float
            Mean molecular mass in grams
        Tmin : float, optional
            Minimum temperature to search
        Tmax : float, optional
            Maximum temperature to search
            
        Returns
        -------
        float
            Minimum cooling time
        """
        from scipy.optimize import minimize_scalar
        
        pressure_value = (rho * ut.constants.kb * T) / mbar_val

        def cooling_function(T_val, P):
            from cooling import cooling_table_logT_cgs, cooling_table_logLambda_cgs
            log_lambda = np.interp(np.log10(T_val), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
            Lambda = 10**log_lambda
            return ut.constants.kb**2 * T_val**2 / abs(P * Lambda)
        
        result = minimize_scalar(
            cooling_function, 
            bounds=(Tmin, Tmax), 
            args=(pressure_value,), 
            method='bounded'
        )
        if result.success:
            return result.fun
        else:
            raise RuntimeError("Minimization did not converge.")
        
    def compute_restart_cooling_time(self):
        """Compute cooling time for restart conditions."""
        import re

        baseDir = self.filename.rsplit('/strat', 1)[0]
        slurm_file = os.path.join(baseDir, 'slurm')

        text = open(slurm_file).read()

        tlim = re.search(r"parthenon/time/tlim=([0-9.eE+-]+)", text)
        lambda_units = re.search(r"cooling/lambda_units_cgs=([0-9.eE+-]+)", text)
        t_cool_restart = get_t_cool_cgs(self.cloud_rho, self.T_cloud, self.mbar) * float(lambda_units.group(1)) * ut.constants.s_to_Myrs
        print(f"Cooling time at restart conditions: {t_cool_restart:.3e} Myrs")
        print(self.mach * self._get_c_s(self.T_base)/1e5)
        print(f"tcool,mix / tcc at restart conditions: {get_t_cool_cgs(np.sqrt(self.env_rho * self.cloud_rho), np.sqrt(self.T_base * self.T_cloud), self.mbar)*float(lambda_units.group(1)) / (self.r_cloud_inserted * self.code_length_cgs * self.chi**0.5 /(self.mach * self._get_c_s(self.T_base))):.3e}")
        lshatter = get_l_shatter(self.env_rho/self.mbar * ut.constants.kb *1e6)[0] * float(lambda_units.group(1)) / self.code_length_cgs
        print(f"  Cloud / Lshatter: {self.r_cloud_inserted * self.code_length_cgs / get_l_shatter(self.env_rho / self.mbar * ut.constants.kb * self.T_base)[0]/ float(lambda_units.group(1)):.3e} ")
        print(f"lshatter at restart conditions: {lshatter:.3e} pc")
        return t_cool_restart
    
    def set_y(self, height):
        """
        Re-center x2min and x2max so that the mesh center is at `height`,
        keeping the cell size (dx2) consistent.

        Parameters
        ----------
        height : float
            Desired center of the x2 direction.
        """
        xmin2 = float(self.reader.get('parthenon/mesh', 'x2min'))
        xmax2 = float(self.reader.get('parthenon/mesh', 'x2max'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))

        # Compute original cell size
        dy = (xmax2 - xmin2) / nx2

        # Compute total domain width from nx2 and dy (ensures dy is exact)
        total_width = nx2 * dy
        half_width = total_width / 2.0

        # Set new min/max centered on `height`
        new_x2min = height - half_width
        new_x2max = height + half_width

        # Update the mesh
        self.reader.set_('parthenon/mesh', 'x2min', new_x2min)
        self.reader.set_('parthenon/mesh', 'x2max', new_x2max)
        self.reader.set_('parthenon/mesh', 'x1min', -half_width)
        self.reader.set_('parthenon/mesh', 'x1max', +half_width)
        self.reader.set_('parthenon/mesh', 'x3min', -half_width)
        self.reader.set_('parthenon/mesh', 'x3max', +half_width)
        self.reader.set_('problem/stratified_box', 'loc_cloud_inserted', [0, height, 0])

        self._enforce_cartesian_grid()

    def radius(self, radius):
        """
        Set cloud radius in pc.
        
        """
        self.reader.set_('problem/stratified_box', 'r_cloud_inserted', radius)
        self.reader.save()
        
