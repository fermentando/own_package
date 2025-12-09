import re
import numpy as np
import scipy.integrate
import scipy.optimize

import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as C

import cooling as mycool


from profiles import mw_profile_g, mw_profile_salem_n, mw_profile_salem_T, halo_profile_density, halo_profile_g, halo_profile_temperature

tcool = mycool.get_t_cool_cgs

class FallingCloud(object):
    """
    Units:
    times       --> Myr
    r_cl        --> pc
    distances   --> kpc
    n           --> cm^-3
    g           --> m/s^2
    velocities  --> km/s
    masses      --> kg
    Integration all done in SI!

    """
    def __init__(self, d0, r_cl0, T_cl = 1e4, profile = 'halo'):

        self.mbar = 1.6 * C.m_p.to('g').value
        self.d0 = d0
        self.r_cl0 = r_cl0
        self.T_cl = T_cl
        self.cs_cold = np.sqrt((5/3. * 1e4 * u.K * C.k_B) / (1.6 * C.m_p)).to('km/s').value

        if profile == 'halo':
            print("Using Halo profile")
            self.profile_rho = halo_profile_density
            self.profile_g = halo_profile_g
            self.profile_T = halo_profile_temperature
        elif profile == 'salem':
            print("Using Salem MW profile")
            self.profile_rho = lambda d: mw_profile_salem_n(d) * self.mbar
            self.profile_g = mw_profile_g
            self.profile_T = mw_profile_salem_T
        else:
            raise ValueError("Unknown mode")

        self.mass0 = (4/3. * (r_cl0 * u.parsec.to('cm'))**3 * np.pi * self.chi(d0) * self.profile_rho(d0))  / u.Msun.to('g')
        # ensure tcool(...) is treated as seconds by astropy, then store as Myr (float)
        self.tcool0 = (tcool(self.profile_rho(d0) * self.chi(d0), self.T_cl, self.mbar) / u.Myr.to('s'))
        # rescale to typical parameters
        Lturb = r_cl0
        f_A = 0.35
        self.tgrow0 = 50 * (f_A / 0.25) * (self.cs_cold / 15.)**(-0.75) * (self.chi(d0)/100.) * (self.r_cl0 / 100.) *\
            (Lturb / 100.)**(-0.25) * (self.tcool0 / 0.03)**(0.25)

    def chi(self, d):
        return self.profile_T(d) / self.T_cl

    def rcl(self, d):
        return self.r_cl0 * (self.profile_P(self.d0) / self.profile_P(d))**(1/3.)

    def tgrow_over_tcc(self, d, rescale_rcl = True):
        """Survival criterion. Uses v ~ g tgrow
        If `rescale_rcl` is True used pressure to change cloud size
        """
        chi = self.chi(d)
        n_cl = chi * self.profile_rho(d) / self.mbar
        g = self.profile_g(d)
        if rescale_rcl:
            rcl = self.rcl(d)
        else:
            rcl = self.r_cl0
        return 8 * (n_cl/1e-2)**(-5/16.) * (rcl / 100.)**(-1/16.) * (1e2 * g/1e-8)**(1/4.) * (chi/100)**(3/4.)

    def get_survival_radius(self, f_S_crit = 4, rescale_rcl=False):
        fun = lambda x : self.tgrow_over_tcc(x,rescale_rcl=rescale_rcl) - f_S_crit
        r = scipy.optimize.root_scalar(fun, x0 = 20, x1= 10)
        assert r.converged, r
        return r.root


    def profile_P(self, d):
        return self.profile_T(d) * self.profile_rho(d) * C.k_B.to('erg/K') / self.mbar


    def integrate(self,stop_value, f_KH = 1., alpha = 5/6., verbose = 1, stop_mode = 'time'):
        """
        Parameters:
            stop_value  -- Depends on `stop_mode` what it can be
            stop_mode   -- Can be:
                             'time' for stopping time in Myr
                             'height' for falling until height (in kpc) above disk
                             'tgrow_over_tcc' for falling until this value is rached
            f_KH        -- 1 for stratified and 5 for constant background recommended
            alpha       -- how does area change with mass change
        """
        def _odefun(t, y):
            # separate variables        
            z, mom, m = y
            v = mom / m
            z_kpc = z / u.kpc.to('m')
            rho_h = self.profile_rho(z_kpc) * u.g.to('kg') / (u.cm.to('m'))**3
            g = -self.profile_g(z_kpc) * u.cm.to('m')
            C0 = 0.6 # drag coefficient
            rcl = self.rcl(z_kpc) * u.parsec.to('m')
            Across = np.pi * rcl**2


            # Compute current growth time
            t_cc = self.chi(z_kpc)**0.5 * rcl / np.abs(v)

            #w_KH = np.clip(f_KH * t_cc / (t+1e-15),1., np.inf)
            w_KH = np.clip(f_KH * self.chi(z_kpc)**0.5 * rcl / (u.kpc.to('m') * (self.d0 - z_kpc)),1.,np.inf)


            tcool_cl = tcool(self.profile_rho(z_kpc) * self.chi(z_kpc), self.T_cl, self.mbar) / u.Myr.to('s')
            vrat = 150. / (np.abs(v)/1e3)
            tcoolrat = tcool_cl / self.tcool0
            mrat = m  / (self.mass0 * u.Msun.to('kg'))
            rhorat = self.profile_rho(z_kpc) / self.profile_rho(self.d0)
            
            #tgrow = w_KH * self.tgrow0 * vrat**(3/5.) * tcoolrat**(1/4.) * (mrat / rhorat)**(1-alpha) * u.Myr.to('s')
            tgrow = w_KH * self.tgrow0 * u.Myr.to('s') * vrat**(3/5.) * tcoolrat**(1/4.) * (mrat / rhorat)**(1-alpha)

            dmomdt = m *  g - 0.5 * rho_h * v**2 * C0 * Across
            t
            r = [
                v, #dz/dt = v
                dmomdt,
                m / tgrow
            ]

            return np.array(r) 
        
        y0 = (self.d0 * u.kpc.to('m'), 1e-3, self.mass0 * u.M_sun.to('kg'))

        tol = (1e-5 * u.kpc.to('m'), 1e-5 * u.Msun.to('kg') * 1e-3 * 1e3, 1e-5 * u.Msun.to('kg'))
        o = dict(first_step = 1e-5 * u.Myr.to('s'), atol = tol, max_step = 1 * u.Myr.to('s'))
        if stop_mode == 'tgrow_over_tcc': # integrate until value of t_grow / t_cc is reached
            def _odestop(t, y): 
                return self.tgrow_over_tcc(y[0] / u.kpc.to('m')) + stop_value
            _odestop.terminal = True
            tend = 1e4 * u.Myr.to('s')
            o['events'] = (_odestop,)
        elif stop_mode == 'time':
            tend = stop_value * u.Myr.to('s')
        elif stop_mode == 'height':
            def _odestop(t, y): 
                return y[0] / u.kpc.to('m') - stop_value
            _odestop.terminal = True
            tend = 1e4 * u.Myr.to('s')
            o['events'] = (_odestop,)
        else:
            raise ValueError("Unknown `stop_mode`")
        r = scipy.integrate.solve_ivp(_odefun, (0,tend),y0, **o)
        if verbose > 0:
            print(r['message'])

        self.integration_result = r
        return r

    def plot_trajectory(self):
        assert self.integration_result is not None, "Have to run integrate first."
        r = self.integration_result
        Myr = u.Myr.to('s')
        plt.plot(r['t'] / Myr, r['y'][0] / u.kpc.to('m'),'-')
        plt.xlabel("t")
        plt.ylabel("z")
        plt.figure()
        plt.plot(r['t'] / Myr, r['y'][1] /r['y'][2] / 1e3)
        plt.xlabel("t")
        plt.ylabel("v")
        plt.figure()
        plt.plot(r['t'] / Myr, r['y'][2] / u.M_sun.to('kg'))
        plt.xlabel("t")
        plt.yscale('log')
        plt.ylabel("Mass (Msun)")

    def plot_timescales(self):
        r = self.integration_result
        time = r['t'][::-1] / u.Myr.to('s')
        plt.plot(time, self.tgrow_over_tcc(r['y'][0] / u.kpc.to('m')))
        plt.axhline(4, c='black', alpha = 0.5)
        plt.xlabel("t (Myr)")
        plt.ylabel("tgrow / tcc")


    def get_tcc_vmax(self, time_limit = np.inf):
        """Returns t_cc in Myr at point of maximum  velocity"""
        r = self.integration_result
        time = r['t'] / u.Myr.to('s')
        mask = time < time_limit
        v = (-r['y'][1] / r['y'][2] / 1e3)[mask]
        vmax = np.max(v)
        dmax = r['y'][0][mask][v == vmax][0] * u.m.to('kpc')
        return self.chi(dmax)**0.5 * self.rcl(dmax) * u.pc.to('km') / vmax * u.s.to('Myr')

    def get_falling_time_to_safety(self, f_S = 4, return_units = 'Myr'):
        """Returns time until t_grow / t_cc(v=t_grow g) = f_S
        """
        r = self.integration_result
        trat = self.tgrow_over_tcc(r['y'][0] / u.kpc.to('m'))
        if trat[0] < f_S:
            return 0.

        # Have to invert as interp expects increasing values
        falltime = np.interp(f_S,trat[::-1],r['t'][::-1] / u.Myr.to('s'))

        if falltime == 0:
            return 0.

        if return_units == 'Myr':
            return falltime
        elif return_units == 't_cc_vmax':
            t_cc = self.get_tcc_vmax(time_limit=falltime)
            return falltime / t_cc