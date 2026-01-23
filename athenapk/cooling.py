"""
Cooling module for AthenaK simulations.

Handles cooling table loading and all cooling-related physics calculations.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import utils as ut
import astropy.constants as C

# Global cooling table variables
cooling_table_logT_cgs = None
cooling_table_logLambda_cgs = None

# Global physics constants (set during module initialization)
gamma = None
mbar = None
mu = None
m_H = None


def initialize_cooling_constants(gamma_val, mbar_val, mu_val, m_H_val):
    """
    Initialize global constants used in cooling calculations.
    
    Parameters
    ----------
    gamma_val : float
        Heat capacity ratio
    mbar_val : float
        Mean molecular mass in grams
    """
    global gamma, mbar, mu, m_H
    gamma = gamma_val
    mbar = mbar_val
    mu = mu_val
    m_H = m_H_val


def load_cooling_table(cooling_table_path):
    """
    Load cooling table from file.
    
    Parameters
    ----------
    cooling_table_path : str
        Path to the cooling table file (two columns: logT, logLambda)
        
    Raises
    ------
    FileNotFoundError
        If cooling table file is not found
    """
    global cooling_table_logT_cgs, cooling_table_logLambda_cgs
    
    try:
        data = np.loadtxt(cooling_table_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cooling table not found: {cooling_table_path}")
    
    cooling_table_logT_cgs = data[:, 0]
    cooling_table_logLambda_cgs = data[:, 1]


def get_c_s(T):
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
    return np.sqrt(5/3. * ut.constants.kb * T / mbar)


def get_t_cool_cgs(rho, T, mbar_val=None, gamma = 5/3.):
    """
    Calculate cooling time in CGS units.
    
    Parameters
    ----------
    rho : float or array
        Density in g/cm^3
    T : float or array
        Temperature in Kelvin
    mbar_val : float, optional
        Mean molecular mass in grams. If None, uses global mbar.
        
    Returns
    -------
    float or array
        Cooling time in seconds
    """
    if mbar_val is None:
        mbar_val = mbar
    
    e = ut.constants.kb * T / (gamma - 1) / mbar_val
    log_lambda = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
    Lambda = 10**log_lambda
    n_H = rho / (m_H * ut.constants.mh)
    return rho * e / (n_H**2 * Lambda)


def get_t_cool_n(T, n, mbar=None):
    """
    Calculate cooling time given number density.
    
    Parameters
    ----------
    n : float or array
        Number density in cm^-3
    T : float or array
        Temperature in Kelvin
        
    Returns
    -------
    float or array
        Cooling time in seconds
    """
    gamma = 5/3.
    if mbar is None:
        mbar = 0.7 * C.m_p.to('g').value
    rho = n * mbar
    e = C.k_B.cgs.value * T / (gamma - 1) / mbar  # erg / g
    log_lambda = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
    Lambda = 10**log_lambda
    n_H = rho / (m_H * ut.constants.mh)
    return rho * e / (n_H**2 * Lambda)


def get_t_cool_min(rho, T, mbar_val, Tmin=1e4, Tmax=1e6):
    """
    Find the minimum cooling time over a temperature range at fixed pressure.
    
    Parameters
    ----------
    rho : float or array-like
        Density in g/cm^3
    T : float or array-like
        Reference temperature in Kelvin (used to compute pressure)
    mbar_val : float
        Mean molecular mass in grams
    Tmin : float, optional
        Minimum temperature to search (default 1e4 K)
    Tmax : float, optional
        Maximum temperature to search (default 1e6 K)
    
    Returns
    -------
    float or ndarray
        Minimum cooling time in seconds
    """
    # Convert inputs to arrays
    rho = np.atleast_1d(rho)
    T = np.atleast_1d(T)
    
    # Check if inputs are scalars
    scalar_input = (rho.size == 1 and T.size == 1)
    
    # Broadcast to common shape
    rho, T = np.broadcast_arrays(rho, T)
    
    # Initialize output array
    results = np.zeros(rho.shape)
    
    def cooling_function(T_val, pressure_value):
        return (ut.constants.kb**2 * T_val**2 / 
                abs(pressure_value * 10**np.interp(np.log10(T_val), 
                                                     cooling_table_logT_cgs, 
                                                     cooling_table_logLambda_cgs)))
    
    # Loop over all elements
    for idx in np.ndindex(rho.shape):
        pressure_value = (rho[idx] * ut.constants.kb * T[idx]) / mbar_val
        
        result = minimize_scalar(
            cooling_function, 
            bounds=(Tmin, Tmax), 
            args=(pressure_value,), 
            method='bounded'
        )
        
        if result.success:
            results[idx] = cooling_function(result.x, pressure_value)
        else:
            raise RuntimeError(f"Minimization did not converge at index {idx}")
    
    # Return scalar if input was scalar
    return results.item() if scalar_input else results

def get_l_shatter(P,mbar=None):
    """
    Calculate the shattering length scale.
    
    Parameters
    ----------
    P : float
        Pressure in dyne/cm^2
        
    Returns
    -------
    tuple
        (shattering_length, temperature_at_minimum)
    """
    def l_shatter_func(T):
        return get_c_s(T) * get_t_cool_n(T, P / (ut.constants.kb * T), mbar)

    res = minimize_scalar(l_shatter_func, bounds=(1e4, 1e6), method='bounded')
    return res.fun, res.x


def plot_l_cool_min_vs_pressure(pressures, mbar_val, Tmin=1e4, Tmax=1e6, show=True):
    """
    Plot minimum cooling length as a function of pressure.
    
    For each pressure (in cgs: dyn/cm^2) find the temperature T in [Tmin, Tmax]
    that minimizes the cooling length and plot it.
    
    Parameters
    ----------
    pressures : float or array
        Pressure values in dyn/cm^2
    mbar_val : float
        Mean molecular mass in grams
    Tmin : float, optional
        Minimum temperature (default 1e4 K)
    Tmax : float, optional
        Maximum temperature (default 1e6 K)
    show : bool, optional
        Whether to display and save the plot (default True)
        
    Returns
    -------
    tuple
        (pressures_array, tmin_array, T_at_min_array)
    """
    pressures = np.atleast_1d(pressures).astype(float)
    tmins = np.empty_like(pressures)
    Tmins = np.empty_like(pressures)

    def cooling_time_at_fixed_pressure(T, P):
        log_lambda = np.interp(np.log10(T), cooling_table_logT_cgs, cooling_table_logLambda_cgs)
        Lambda = 10.0 ** log_lambda
        return (ut.constants.kb ** 2 * T ** 2) / (abs(P) * Lambda)

    for i, P in enumerate(pressures):
        res = minimize_scalar(lambda T: cooling_time_at_fixed_pressure(T, P),
                                bounds=(Tmin, Tmax), method='bounded')
        if not res.success:
            raise RuntimeError(f"Minimization failed for pressure={P}")
        Tmins[i] = res.x
        tmins[i] = res.fun

    # Plot (log-log is usually informative)
    if show:
        fig, ax = plt.subplots()
        ax.loglog(pressures, get_c_s(Tmins) * tmins / ut.constants.pc_to_cm, marker='o', lw=1)
        ax.set_xlabel("Pressure (dyn cm$^{-2}$)")
        ax.set_ylabel("l_cool,min (pc)")
        ax.grid(which='both', ls=':')
        # optional twin axis to show T_at_min
        ax2 = ax.twinx()
        ax2.semilogx(pressures, Tmins, color='tab:orange', marker='x', lw=0.5)
        ax2.set_ylabel("T at minimum (K)", color='tab:orange')
        for tl in ax2.get_yticklabels():
            tl.set_color('tab:orange')
        plt.axvline(1e-27 * ut.constants.kb * 1e6 / 0.6 / mbar_val, color='gray', ls='--', lw=1)
        plt.tight_layout()
        plt.savefig("l_cool_min_vs_pressure.png", dpi=300)

    return pressures, tmins, Tmins
