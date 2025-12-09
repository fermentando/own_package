import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from cooling import get_t_cool_n, load_cooling_table
import astropy.units as u
import astropy.constants as C
import latexify
import os
from flow_props import vel_evolution, mass_evolution
from stratified_box import StratifiedBox

# Load cooling table
load_cooling_table("/viper/ptmp/ferhi/cooling_tables/gnat-sternberg.cooling_1Z")

# Unit conversions
pc   = 3.086e18
Myr  = 3.15576e13
Msun = 1.989e33
km   = 1e5
mbar = 9.8e-25

def strat_n(r):
    n0 = 1
    H = 4
    a = 0.001

    return n0 * np.exp(-a * ((1 + (r/a/H)**2)**0.5 -1))

def strat_g(r):
    g0 = 1e-8
    a = 0.001
    H = 4
    return g0 * r/a/H /(1 + (r/a/H)**2)**0.5 

def strat_T(r):
    return np.ones_like(r) * 1e6


class TurbFallingCloud(object):
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
    def __init__(self, d0, r_cl0, T_cl = 1e4, vturb = 15, profile = 'MW', tcool = None, v0 = 0):
        self.d0 = d0 #kpc
        self.r_cl0 = r_cl0 #pc
        self.T_cl = T_cl
        self.vturb = vturb * 1e3
        self.v0 = v0 * 1e3
        if profile == 'MW':
            self.profile_n = strat_n     # cm^-3
            self.profile_g = strat_g     # (same units used elsewhere)
            self.profile_T = strat_T      # K
            
        self.mass0 = 4/3. * (r_cl0 * u.parsec.to('cm') )**3 * np.pi * self.chi(d0) * self.profile_n(d0) * mbar / u.M_sun.to('g')
        if tcool == None:  
            self.Lambda_units = 1
            self.tcool0 = get_t_cool_n(T_cl, self.profile_n(d0) * self.chi(d0)) / u.Myr.to('s')
        else:
            self.Lambda_units = get_t_cool_n(T_cl, self.profile_n(d0) * self.chi(d0)) / u.Myr.to('s') / tcool
            self.tcool0 = tcool
        Lturb = r_cl0
        f_A = 0.23
        if self.vturb / 1000 > 30: f_A *= 3.3
        self.tgrow0 = 12 * (f_A / 0.23)  * (self.chi(d0)/100.) * (self.r_cl0 / 100.) *\
            (Lturb / 100.)**(-0.25) * (self.tcool0 / 0.03 )**(0.25)

    def chi(self, d):
        return self.profile_T(d) / self.T_cl

    def rcl(self, d):
        return self.r_cl0 * (self.profile_P(self.d0) / self.profile_P(d))**(1/3.)
        

    def profile_P(self, d):
        return self.profile_T(d) * self.profile_n(d)


    def integrate(self,stop_value, f_KH = 1, alpha = 5/6., f_T = 1, verbose = 1, stop_mode = 'time', dense_output=True, drag_bool = True):
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
            z, v, m = y
            z_kpc = z / u.kpc.to('m') 
            rho_h = self.profile_n(z_kpc) * mbar * u.g.to('kg') / (u.cm.to('m')**3)
            g = -self.profile_g(z_kpc) * 1e-2  
            C0 = 0.6 # drag coefficient
            rcl = self.rcl(z_kpc) 
            rcl_m = rcl * u.parsec.to('m')
            Across = np.pi * rcl_m**2 


            r_pc_to_kpc = u.parsec.to('kpc')
            w_KH = f_KH * np.sqrt(self.chi(z_kpc)) * \
              (self.r_cl0 * r_pc_to_kpc) / \
              abs(self.d0 - z_kpc)


            tcool_cl = get_t_cool_n(self.T_cl, self.profile_n(z_kpc) * self.chi(z_kpc)) / u.Myr.to('s') / self.Lambda_units
            if drag_bool:
                v_tot = self.vturb + (2/(1 + 2*w_KH**2)) * v
                vrat = 150. / min(abs(v_tot)/1000, 150.)
            else:
                vrat = 150. / min(abs(v)/1000, 150.)

            tcoolrat = tcool_cl / self.tcool0
            mrat = m  / (self.mass0 * u.Msun.to('kg'))
            rhorat = self.profile_n(z_kpc) / self.profile_n(self.d0)

            tgrow = self.tgrow0 * u.Myr.to('s') * vrat**(3/5.) * tcoolrat**(1/4.) * (mrat / rhorat)**(1-alpha)

            

            def drag(self, rho_h, v):
                if drag_bool: turb_drag = 5 * rho_h * self.vturb**2 * Across + 0.5 * rho_h * (v) **2 * C0 * Across 
                else: turb_drag = 0.5 * rho_h * (v) **2 * C0 * Across 
                expr = g +turb_drag / m
                if expr > 0.: return expr
                else: return expr

            dzdt = v
            dvdt = drag(self = self, rho_h=rho_h, v= v) - ((v)  /(tgrow))
            dmdt = m / tgrow 
            r = [dzdt, dvdt, dmdt]

            return np.array(r) 
        
        y0 = (self.d0 * u.kpc.to('m'), self.v0, self.mass0 * u.M_sun.to('kg'))

        tol = (1e-5 * u.kpc.to('m'), abs(self.v0) * 1e-5, 1e-5 * self.mass0* u.M_sun.to('kg'))
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
        r = scipy.integrate.solve_ivp(_odefun, (0,tend),y0, **o, dense_output=dense_output)
        if verbose > 0:
            print(r['message'])

        self.integration_result = r
        return r
    
    def vTgrow(self, z,v,m, alpha=5/6, f_T = 1.):
        z_kpc = z / u.kpc.to('m')
        w_KH = np.maximum(1, f_T * np.sqrt(self.chi(z_kpc)) *
            (self.r_cl0 * u.pc.to('kpc')) /
            abs(self.d0 - z_kpc))


        tcool_cl = get_t_cool_n(self.T_cl, self.profile_n(z_kpc) * self.chi(z_kpc)) / u.Myr.to('s') / self.Lambda_units
        vrat = np.ones_like(z_kpc)
        v_turb  = abs(self.vturb)/1000.
        v_infall = abs(v)/1000.


        v_tot = v_turb + (2/(1 + w_KH**2)) * v_infall
        vrat = 150. / np.minimum(abs(v_tot), 150.)

        # Result array:
        #vrat = np.where(
        #    w_KH > 1,
        #    150. / v_turb,
        #    150. / v_infall
        #)
        tcoolrat = tcool_cl / self.tcool0
        mrat = m  / (self.mass0 * u.Msun.to('kg'))
        rhorat = self.profile_n(z_kpc) / self.profile_n(self.d0)

        tgrow = self.tgrow0 * u.Myr.to('s') * vrat**(3/5.) * tcoolrat**(1/4.) * (mrat / rhorat)**(1-alpha)
        return self.profile_g(z_kpc) * tgrow / 1e2
    
    def vTdrag(self, z,m):
        z_kpc = z / u.kpc.to('m')
        rcl = self.rcl(z_kpc) * u.pc.to('m')  # Also fix: use z_kpc
        rho_h = self.profile_n(z_kpc) * mbar * u.g.to('kg') / (u.cm.to('m')**3)  # Convert to kg/m³
        #return np.sqrt(2 * m * self.profile_g(z_kpc) * 1e-2 / (rho_h * np.pi * rcl**2 * 0.6)) - self.vturb / np.sqrt(0.6)
        return np.ones_like(z) * (np.sqrt(2 * m * self.profile_g(z_kpc) * 1e-2 / (rho_h * np.pi * rcl**2 * 0.6))) - 2 * self.vturb

        #return np.ones_like(z) * rho_h * self.vturb**2 * np.pi * self.r_cl0**2 

    def plot_trajectory(self):
        assert self.integration_result is not None, "Have to run integrate first."
        r = self.integration_result
        Myr = u.Myr.to('s')

        latexify(columns=1)
        plt.plot(r['t'] / Myr, r['y'][0] / u.kpc.to('m'),'-')
        plt.xlabel("t")
        plt.ylabel("z")
        plt.figure()
        plt.plot(r['t'] / Myr, -r['y'][1] / 1e3)
        plt.ylim(-10,10)
        plt.plot(r['t'] / Myr, self.vTgrow(r['y'][0], r['y'][1], r['y'][2])/1e3, color='red')
        plt.plot(r['t'] / Myr, self.vTdrag(r['y'][0],  r['y'][2])/1e3, color='black')
        plt.plot(r['t'] / Myr, 1e-8/1e5 * r['t'], color='green')
        #plt.yscale('log')
        plt.xlabel("t")
        plt.ylabel("v")
        plt.figure()
        plt.plot(r['t'] / Myr, r['y'][2] / r['y'][2][0])
        plt.xlabel("t")
        plt.yscale('log')
        plt.ylabel("Mass (Msun)")


    def plot_timescales(self):
        r = self.integration_result
        time = r['t'][::-1] / u.Myr.to('s')
        plt.plot(time, self.vTgrow(r['y'][0], r['y'][1], r['y'][2]))
        plt.axhline(4, c='black', alpha = 0.5)
        plt.xlabel("t (Myr)")
        plt.ylabel("tgrow / tcc")


if __name__ == "__main__":

    listruns = [
        ('/viper/ptmp/ferhi/StratDisk/Rsys/m0.1/r10/t1e3', 1.83e-5, 7, 10, 12, 8),
        ('/viper/ptmp/ferhi/StratDisk/Rsys/m0.1/r100/t1e6', 1.06e-1, 20, 100, 50, 0), 
        ('/viper/ptmp/ferhi/StratDisk/Rsys/m0.1/r100/t1e4', 1.06e-3, 15, 100, 70, 4),
    ]

    import matplotlib.pyplot as plt
    plt.style.use('custom_plot')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sc_kw = dict(s=22, edgecolor="k", linewidth=0.3, alpha=0.8)

    for i, run in enumerate(listruns):
        path, tcool, vturb, rcl, time, v0 = run
        # create cloud and integrate
        cloud = TurbFallingCloud(d0=3, r_cl0=rcl, vturb=vturb, tcool = tcool, v0=v0)
        try:
            cloud.integrate(time, stop_mode='time', verbose=0, dense_output=False)
        except Exception as e:
            print(f"Integration failed for {os.path.basename(path)}: {e}")
            continue

        r = cloud.integration_result
        t = r['t'] / Myr
        m = r['y'][2] / r['y'][2,0]
        v = -r['y'][1] / 1e3  # km/s

        ax_top = axes[0, i]
        ax_bot = axes[1, i]

        # Top subplot: trajectory z(t)
        ax_top.plot(t, m, '-')
        ax_top.set_title(fr"$r = ${rcl}")
        if i ==0:
            ax_top.set_ylabel(r'm / m$_0$')
            ax_bot.set_ylabel('v (km/s)')
        ax_top.grid(True)

        # Bottom subplot: velocity and analytic timescales/drag
        ax_bot.plot(t, v, '-', label='v(t) [km/s]')
        # overlay vTgrow and vTdrag (converted to km/s similar to original plotting)
        try:
            ax_bot.plot(t, cloud.vTgrow(r['y'][0], r['y'][1], r['y'][2]) / 1e3, color='red', label='vTgrow', linestyle='--', alpha=0.5)
            ax_bot.plot(t, cloud.vTdrag(r['y'][0], r['y'][2]) / 1e3, color='black', label='vTdrag', linestyle='--', alpha=0.5)
            ax_bot.plot(t, 1e-8/1e5 * r['t'], color='green', linestyle='--', alpha=0.5)
        except Exception:
            # if analytic functions expect different shapes or fail, ignore overlays
            pass


        ax_bot.set_xlabel('t (Myr)')
        ax_bot.grid(True)
        #if i == 2:
        #    ax_bot.legend(fontsize='small')


        # Compare with sim
        sim = StratifiedBox(os.path.join(path, 'strat.in'), dir=path)
        times, norm_mass, cgout, wgout, total = mass_evolution(path, gout=True)
        tvs, vel = vel_evolution(path)
        mask = ~np.isnan(norm_mass)
        timeseries = times/sim.t_eddy - 6
        idx_0 = np.argmin(np.abs(timeseries))
        norm_mass = norm_mass- norm_mass[idx_0]

        timeseries = timeseries[mask] * sim.t_eddy * sim.code_times_cgs / u.Myr.to('s')
        norm_mass = 10**norm_mass[mask]
        ax_bot.scatter(timeseries, vel * sim.code_length_cgs / sim.code_times_cgs / 1e5, color='orange', **sc_kw)
        ax_top.scatter(timeseries, norm_mass, color='orange', **sc_kw)

        # Format
        ax_top.set_yscale('log')
        ax_top.set_ylim(-0.8, 80)
        if i ==1: ax_bot.set_ylim(-10, 100)
        if i ==0: ax_bot.set_ylim(-10,50)
        if i ==2: ax_bot.set_ylim(0, 80)

    plt.tight_layout()
    print("Saving figure to: '/u/ferhi/own_package/plots/multiplot_evolution.png' ")
    plt.savefig('/u/ferhi/own_package/plots/multiplot_evolution.png')
