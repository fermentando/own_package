"""
Physics utility functions for AthenaK simulations.

Contains general physics calculations and helper functions.
"""

import math
import numpy as np
import utils as ut


def calculate_pressure(T, rho, mbar):
    """
    Calculate pressure from temperature and density.
    
    Parameters
    ----------
    T : float
        Temperature in Kelvin
    rho : float
        Density in g/cm^3
    mbar : float
        Mean molecular mass in grams
        
    Returns
    -------
    float
        Pressure in dyne/cm^2
    """
    k_B = 1.3807e-16  # erg/K
    return (rho * k_B * T) / mbar


def calculate_v_wind(mach, gamma, pressure, rho_amb):
    """
    Calculate wind velocity from Mach number and ambient conditions.
    
    Uses jump conditions across the shock.
    
    Parameters
    ----------
    mach : float
        Mach number
    gamma : float
        Heat capacity ratio
    pressure : float
        Pressure in dyne/cm^2
    rho_amb : float
        Ambient density in g/cm^3
        
    Returns
    -------
    float
        Wind velocity in cm/s
    """
    jump3 = 2 * (1 - 1 / mach**2) / (gamma + 1)
    velocity_of_sound = math.sqrt(gamma * pressure / rho_amb)
    return jump3 * mach * velocity_of_sound


def estimate_mach_from_v_wind(v_wind_desired, gamma, pressure, rho_amb, 
                               mach_guess=1.0, tolerance=1e-6, max_iter=100):
    """
    Estimate Mach number from desired wind velocity.
    
    Uses Newton-Raphson iteration to find the Mach number that produces
    the desired wind velocity.
    
    Parameters
    ----------
    v_wind_desired : float
        Desired wind velocity in cm/s
    gamma : float
        Heat capacity ratio
    pressure : float
        Pressure in dyne/cm^2
    rho_amb : float
        Ambient density in g/cm^3
    mach_guess : float, optional
        Initial guess for Mach number (default 1.0)
    tolerance : float, optional
        Convergence tolerance (default 1e-6)
    max_iter : int, optional
        Maximum iterations (default 100)
        
    Returns
    -------
    float
        Estimated Mach number
    """
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
