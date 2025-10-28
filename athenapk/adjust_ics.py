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
        self.mbar = mbar

    def _load_simulation_parameters(self):
        self.R_cloud = float(self.reader.get('problem/wtopenrun', 'r0_cgs'))
        self.rho_cloud = float(self.reader.get('problem/wtopenrun', 'rho_cloud_cgs'))
        self.rho_wind = float(self.reader.get('problem/wtopenrun', 'rho_wind_cgs'))
        self.T_wind = float(self.reader.get('problem/wtopenrun', 'T_wind_cgs'))
        self.T_cloud = self.T_wind * self.rho_wind / self.rho_cloud
        self.v_wind = self._get_wind_velocity()
        self.n_mix = np.sqrt(self.rho_wind * self.rho_cloud) / mbar

    def _get_wind_velocity(self):
        try:
            return float(self.reader.get('problem/wtopenrun', 'v_wind_cgs'))
        except:
            try:
                Mach_wind = float(self.reader.get('problem/wtopenrun', 'mach_wind'))
            except:
                Mach_wind = float(self.reader.get('problem/wtopenrun', 'Mach_wind'))
            return np.sqrt(gamma * ut.constants.kb * self.T_wind / mbar) * Mach_wind
    
    def _modify_shock_mach(self):

        pressure = calculate_pressure(self.T_wind, self.rho_wind, mbar = mbar)
        mach_est = estimate_mach_from_v_wind(self.v_wind, gamma, pressure, self.rho_wind)
        self.reader.set_('problem/wtopenrun', 'mach_shock', mach_est)
        self.reader.save()
        print('Mach shock: ', mach_est)


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
        lamba = {1/self.n_mix/get_t_cool(self.rho_cloud/self.mbar, self.T_cloud) * ut.constants.kb * 1e4:.3g} erg cm^3/s
        r_crit = {self.tcoolmix * self.v_wind / 10 :.3g} cm
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
        rescaled_size = 1/10 * R_internal_units / resol_factor / cell_size
        
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
    
# --------------------
# Stratified box generation functions
# --------------------

class StratifiedBox:
    def __init__(self, filename_input, dir):
        self.filename = filename_input
        self.dir = dir
        self.reader = ut.AthenaPKInputFileReader(filename_input)
        type_box = str(self.reader.get('job', 'problem_id'))
        self._initialize_constants()
        self._load_simulation_parameters()
        self._load_cooling_table(dir)
        self._cloud_conditions()
        self._enforce_cartesian_grid()
        if 'simple' not in type_box: self._set_t_corr()
        self._timescales()


    def _initialize_constants(self):
        global gamma, mbar, mu_H
        gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        mu_H = 1.0
        mu = 1 / (He_mass_fraction * 3 / 4 + (1 - He_mass_fraction) * 2)
        mbar = mu * ut.constants.uam
        self.mbar = mbar

    def _load_simulation_parameters(self):  
        self.surface_density= float(self.reader.get('problem/stratified_box', 'surface_density'))
        self.T_base = float(self.reader.get('problem/stratified_box', 'T_base'))
        self.a_over_H = float(self.reader.get('problem/stratified_box', 'a_over_H'))
        self.r_cloud_inserted = float(self.reader.get('problem/stratified_box', 'r_cloud_inserted'))
        self.T_cloud = float(self.reader.get('problem/stratified_box', 'T_cloud'))
        self.chi = self.T_base / self.T_cloud

        self.code_mass_cgs = float(self.reader.get('units', 'code_mass_cgs'))
        self.code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        self.code_times_cgs = float(self.reader.get('units', 'code_time_cgs'))





    def _load_cooling_table(self, dir):
        rel_path = self.reader.get('cooling', 'table_filename')
        cooling_table_path = os.path.abspath(os.path.join(dir, rel_path))
        try:
            data = np.loadtxt(cooling_table_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cooling table not found: {cooling_table_path}")
        global cooling_table_logT_cgs, cooling_table_logLambda_cgs
        cooling_table_logT_cgs, cooling_table_logLambda_cgs = data[:, 0], data[:, 1]

    def _cloud_conditions(self):
        mbar_over_kb = self.mbar/ut.constants.kb 

        self.g = 2 * np.pi * ut.constants.G * self.surface_density * self.code_mass_cgs / (self.code_length_cgs)**2
        c_s = np.sqrt(self.T_base / mbar_over_kb)
        self.H = c_s**2/ self.g
        rho_base = (self.surface_density * self.code_mass_cgs / (self.code_length_cgs)**2) / (2 * self.H)

        self.y_centre = float(self.reader.get('parthenon/mesh', 'x2max')) - self.r_cloud_inserted * 10
        self.env_rho = rho_base * np.exp(-self.a_over_H * (np.sqrt(1 + (self.y_centre * self.code_length_cgs / (self.a_over_H * self.H))**2) - 1))
        self.cloud_rho = self.chi * self.env_rho

    def _timescales(self):
        Myr = 3.154e13            # seconds
        pc = 3.086e18  
        g_fid = 1e-8               # cm/s^2
        r_100 = 100.0              # reference radius (unit depends on your context)
        c_s_150 = 150e5 

        self.t_cool_cl = get_t_cool_cgs(self.cloud_rho, self.T_cloud, self.mbar)
        self.t_grow_subsonic = (
            40 * Myr
            * (self.g / g_fid)**(-3/8)
            * (self.chi / 100)**(5/8)
            * (self.r_cloud_inserted / r_100)**(15/32)
            * ( self.t_cool_cl/ (0.03 * Myr))**(5/32)
        )

        self.t_grow_supersonic = (
            35 * Myr
            * (get_c_s(self.T_base) / c_s_150)**(-3/5)
            * (self.chi / 100)
            * (self.r_cloud_inserted / r_100)**(3/4)
            * (self.t_cool_cl / (0.03 * Myr))**(1/4)
        )

        self.t_ff = get_c_s(self.T_base) / self.g

        self.r_sonic = (
            150 * pc
            * (self.t_cool_cl / (0.03 * Myr))**(-1/3)
            * (self.g / g_fid)**(-4/3)
            * (self.chi / 100)**(-4/3)
            * (get_c_s(self.T_base) / c_s_150)**(32/15)
        )

        self.r_ss = (
            100 * pc
            * (self.t_cool_cl / (0.03 * Myr))**(-1)
            * (self.g / g_fid)**(-2)
            * (self.chi / 100)**(-2)
            * (get_c_s(self.T_base) / c_s_150)**(12/5)
        )

        self.r_ratio = (
            1.5
            * (self.t_cool_cl / (0.03 * Myr))**(2/3)
            * (self.g / g_fid)**(2/3)
            * (self.chi / 100)**(2/3)
            * (get_c_s(self.T_base)  / c_s_150)**(-4/15)
        )

        self.t_surv_sub = (5e-3 * Myr / self.t_cool_cl) * (self.r_cloud_inserted / 100 ) **(1/5) * (self.g / g_fid) **(-4/5) * (self.chi / 100 ) **(-12/5)

        self.v_drag = np.sqrt( 2 * self.chi * self.r_cloud_inserted *  ut.constants.pc_to_cm * self.g / 0.47 )
        
        self.v_grow_sub = self.g * self.t_grow_subsonic

        self.t_cc = self.r_cloud_inserted * ut.constants.pc_to_cm * self.chi**0.5 / get_c_s(self.T_base)

        
    
    def stateICs(self):
        nx2 = float(self.reader.get('parthenon/mesh', 'nx2'))
        x2min, x2max = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        dcell = (x2max - x2min)/nx2
        print(f"""
        >> Strat disk properties <<

        H = {self.H/ut.constants.pc_to_cm:.3e} pc
        g = {self.g:.3e} cgs
        r_cl / d_cell = {self.r_cloud_inserted /dcell:.3e} 

                     
        R_cl = {self.r_cloud_inserted:.3e} pc
        n_0 = {self.env_rho/self.mbar:.3e} pc
        chi = {self.chi}
        t_cool = {get_t_cool_cgs(self.cloud_rho, 1e4, self.mbar) * ut.constants.s_to_Myrs :.3e} Myrs

        surv_ratio = {get_t_cool_cgs(self.cloud_rho, 1e4, self.mbar) * ut.constants.s_to_Myrs/
                      (5e-3 * (self.r_cloud_inserted/100)**(1/5) * (self.g / 1e-8)**(-4/5) * (self.chi/100)**(-12/5))
                      }

        r_sonic = {self.r_sonic/ut.constants.pc_to_cm:.3e} pc
        r_ss = {self.r_ss/ut.constants.pc_to_cm:.3e} pc
        r_sonic / r_ss = {self.r_ratio:.3e} pc


        fs tcc / tgrow = {self.t_surv_sub:.3e} 

        """)

    def _scale_mesh(self, axis_scaling):
        for axis in ['x1', 'x2', 'x3']:
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}min', axis_scaling)
            self.reader.change_aspect_xlim('parthenon/mesh', f'{axis}max', axis_scaling)
        self.reader.save()
    

    def _enforce_cartesian_grid(self):
        nx1 = int(self.reader.get('parthenon/mesh', 'nx1'))
        nx2 = int(self.reader.get('parthenon/mesh', 'nx2'))
        print(nx2)
        xmin1, xmax1 = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        xmin2, xmax2 = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        cell_size_y = (xmax2 - xmin2) / nx2
        #cell_size_y = (xmax2 - xmin2) / nx2
        y_adjustment = cell_size_y * nx1 - (xmax1 - xmin1)
        for axis in ['1', '3']:
            self.reader.set_('parthenon/mesh', f'x{axis}max', (xmax1 + abs(xmax1)/(xmax1 - xmin1)*y_adjustment))
            self.reader.set_('parthenon/mesh', f'x{axis}min', (xmin1 - abs(xmin1)/(xmax1 - xmin1)*y_adjustment))
            self.reader.save()

    def _set_t_corr(self):
        T0 = float(self.reader.get('problem/stratified_box', 'T_base'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        code_time_cgs = float(self.reader.get('units', 'code_time_cgs'))
        code_velocity_cgs = code_length_cgs / code_time_cgs
        L_box = Lymax - Lymin
        cs = get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach / code_velocity_cgs

        #p0 = 1e-26 * ut.constants.kb * T0 / mbar  # Reference pressure
        #p_floor = p0*0.01

        L_drive = L_box/k_peak
        t_eddy = L_drive/v_turb
        accel_rms  = v_turb**2 / (L_drive) 

        tlim = 10*t_eddy
        dt_hst = 0.001*t_eddy 
        dt_hdf = 0.5*t_eddy 
        dt_rst = 10*t_eddy 
        
        print("this is dt_hdf: ", dt_hdf)
        self.reader.set_('problem/stratified_box', 'rescale_code_time_to_tff', 'false')
        self.reader.set_('problem/turbulence', 'corr_time', t_eddy)
        #self.reader.set_('hydro', 'pfloor', p_floor)
        self.reader.set_('problem/turbulence', 'accel_rms', accel_rms)
        self.reader.set_('parthenon/time', 'tlim', tlim)
        self.reader.set_('parthenon/output1', 'dt', dt_hst)
        self.reader.set_('parthenon/output2', 'dt', dt_hdf)
        self.reader.set_('parthenon/output3', 'dt', dt_rst)
        self.reader.save()
        print(f"Driving correlation time set to {t_eddy:.3e} s")
    
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

    def rescale_to_H(self):
        box_height = 6 * self.H / self.code_length_cgs
        fpos = 0.9; fneg = -0.1
        nx2 = int(self.reader.get('parthenon/meshblock', f'nx2'))
        dim_cell = box_height / nx2
        for axis in [ '2']:
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
        



# --------------------
# Turbulent box generation functions
# --------------------

class TurbulentBox:
    def __init__(self, filename_input, dir):
        self.filename = filename_input
        self.dir = dir
        self.reader = ut.AthenaPKInputFileReader(filename_input)
        self._initialize_constants()
        self._set_t_corr()

    def _initialize_constants(self):
        global gamma, mbar, mu_H
        gamma = float(self.reader.get('hydro', 'gamma'))
        He_mass_fraction = float(self.reader.get('hydro', 'He_mass_fraction'))
        mu_H = 1.0
        mu = 1 / (He_mass_fraction * 3 / 4 + (1 - He_mass_fraction) * 2)
        mbar = mu * ut.constants.uam
        self.mbar = mbar


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

    def _set_t_corr(self):
        T0 = float(self.reader.get('problem/turbulence', 'T0'))
        #rho0 = float(self.reader.get('problem/turbulence', 'rho0'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x1min')), float(self.reader.get('parthenon/mesh', 'x1max'))
        code_length_cgs = float(self.reader.get('units', 'code_length_cgs'))
        code_time_cgs = float(self.reader.get('units', 'code_time_cgs'))
        code_velocity_cgs = code_length_cgs / code_time_cgs
        L_box = Lymax - Lymin
        cs = get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach / code_velocity_cgs

        #p0 = 1e-26 * ut.constants.kb * T0 / mbar  # Reference pressure
        #p_floor = p0*0.01

        L_drive = L_box/k_peak
        t_eddy = L_drive/v_turb
        accel_rms  = v_turb**2 / (L_drive) 

        tlim = 10*t_eddy * code_time_cgs
        dt_hst = 0.0001*t_eddy * code_time_cgs
        dt_hdf = 0.5*t_eddy * code_time_cgs
        dt_rst = 0.5*t_eddy * code_time_cgs
        
        self.reader.set_('problem/turbulence', 'corr_time', t_eddy)
        #self.reader.set_('hydro', 'pfloor', p_floor)
        self.reader.set_('problem/turbulence', 'accel_rms', accel_rms)
        self.reader.set_('parthenon/time', 'tlim', tlim)
        self.reader.set_('parthenon/output1', 'dt', dt_hst)
        self.reader.set_('parthenon/output2', 'dt', dt_hdf)
        self.reader.set_('parthenon/output3', 'dt', dt_rst)
        self.reader.save()
        print(f"Driving correlation time set to {t_eddy:.3e} s")
    
    def _set_t_corr(self):
        T0 = float(self.reader.get('problem/stratified_box', 'T_base'))
        #rho0 = float(self.reader.get('problem/turbulence', 'rho0'))
        mach = float(self.reader.get('problem/turbulence', 'Mach_drive'))
        k_peak = float(self.reader.get('problem/turbulence', 'kpeak'))
        Lymin, Lymax = float(self.reader.get('parthenon/mesh', 'x2min')), float(self.reader.get('parthenon/mesh', 'x2max'))
        L_box = Lymax - Lymin
        cs = get_c_s(T0)  # Sound speed in the medium
        v_turb = cs * mach

        #p0 = rho0 * ut.constants.kb * T0 / mbar  # Reference pressure
        dfloor = 1e-24*0.01

        L_drive = L_box/k_peak
        t_eddy = L_drive/v_turb
        accel_rms  =  v_turb**2 / (L_drive) 

        tlim = 8*t_eddy
        dt_hst = 0.01*t_eddy
        dt_hdf = 0.1*t_eddy
        dt_rst = 0.5*t_eddy

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
    

def get_c_s(T):
    return np.sqrt(gamma * ut.constants.kb * T / mbar)

def get_t_cool_cgs(rho, T, mbar):
    e = ut.constants.kb * T / (gamma - 1) / mbar
    log_lambda = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
    Lambda = 10**log_lambda
    n_H = rho / mbar
    return rho * e / (n_H**2 * Lambda)

def get_t_cool(n, T):
    rho = n * mbar
    e = ut.constants.kb * T / (gamma - 1) / mbar
    log_lambda = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
    Lambda = 10**log_lambda
    n_H = rho / mbar
    return rho * e / (n_H**2 * Lambda)


def get_t_cool_min(rho, T, mbar, Tmin=1e4, Tmax=1e6):
    pressure_value = (rho * ut.constants.kb * T) / mbar 

    def cooling_function(T, pressure_value):
        return ut.constants.kb**2 * T**2 / abs(pressure_value * 10**np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs))
    
    result = minimize_scalar(
        cooling_function, 
        bounds=(Tmin, Tmax), 
        args=(pressure_value), 
        method='bounded'
    )
    if result.success:
        print("This is the temperature at minimum:", result.x)
        return cooling_function(result.x, pressure_value)
    else:
        raise RuntimeError("Minimization did not converge.")
    

def get_l_shatter(P):
    def l_shatter_func(T):
        return get_c_s(T) * get_t_cool(P / (ut.constants.kb * T), T)

    res = minimize_scalar(l_shatter_func, bounds=(1e4, 1e6), method='bounded')
    return res.fun, res.x

import math

def calculate_pressure(T, rho, mbar):
    k_B = 1.3807e-16  # erg/K
    m_p = 1.6726e-24  # g
    return (rho * k_B * T) / (mbar)  # pressure in dyn/cm^2

def calculate_v_wind(mach, gamma, pressure, rho_amb):
    jump3 = 2 * (1 - 1 / mach**2) / (gamma + 1)
    velocity_of_sound = math.sqrt(gamma * pressure / rho_amb)
    return jump3 * mach * velocity_of_sound

def estimate_mach_from_v_wind(v_wind_desired, gamma, pressure, rho_amb):
    mach_guess = 1.0
    tolerance = 1e-6
    max_iter = 100
    for j in range(max_iter):
        current_v = calculate_v_wind(mach_guess, gamma, pressure, rho_amb)
        if abs(current_v - v_wind_desired) < tolerance:
            return mach_guess

        delta = 1e-6
        v_plus = calculate_v_wind(mach_guess + delta, gamma, pressure, rho_amb)
        derivative = (v_plus - current_v) / delta

        if derivative == 0:  # Avoid divide-by-zero
            break

        mach_guess -= (current_v - v_wind_desired) / derivative

        if j == max_iter - 1:
            print("Maximum number of iterations reached")
    return mach_guess


if __name__ == "__main__":
    
    localDir = os.getcwd()
    if os.path.isfile(os.path.join(localDir, "ism.in")):
        sim = SingleCloudCC(os.path.join(localDir, 'ism.in'), dir=localDir)
        command = str.lower(sys.argv[1])
        match command:
            case "check":
                sim._modify_shock_mach()
                sim.state_ICs()
            case "adjust":
                print(float(sys.argv[2]))
                sim.reset_survival(float(sys.argv[2]), 8)
            case "enlarge_y":
                sim.enlarge_dim(increase_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 1,
                            axs=[2])
            case "enlarge_x":
                sim.enlarge_dim(increase_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 1, 
                            axs = [1,3])
            case "res":
                sim.set_rin_res(resol_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 8)
            case "mach_shock":
                sim._modify_shock_mach()
            case _:
                raise ValueError("Invalid choice: pick amongst checking the current survival ratio, 'check', or adjusting to new ratio, 'adjust' followed by your new t_coolmix/t_cc value.")
        

    elif os.path.isfile(os.path.join(localDir, "strat.in")):
        sim = StratifiedBox(os.path.join(localDir, 'strat.in'), dir=localDir)
        command = str.lower(sys.argv[1])
        match command:
            case "check":
                sim._enforce_cartesian_grid() 
                sim.stateICs()         
            case "enlarge_y":
                sim.enlarge_dim(increase_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 1,
                            axs=[2])
            case "enlarge_x":
                sim.enlarge_dim(increase_factor=float(sys.argv[2]) if len(sys.argv) == 3 else 1, 
                            axs = [1,3]) 
            case "rescale":
                sim.rescale_to_H()


