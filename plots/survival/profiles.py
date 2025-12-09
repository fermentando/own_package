import numpy as np
import astropy.constants as C
import astropy.units as u

surface_density_Msun_pc2 =125
H = 4
a_over_H = 0.001


def _get_Rvir(Mvir):
    """Returns Rvir in kpc"""
    rho_crit = 8.5e-27 * 1.478e28
    dens = 200. * rho_crit
    Rvir = (Mvir / (4/3. * np.pi * dens))**(1/3.)
    return Rvir


def nfw_profile_mcum(r, Mvir, c):
    """returns m(<r) in units of Mvir"""
    consts = Mvir / (np.log(1 + c) - c / (1+c))
    Rvir = _get_Rvir(Mvir)
    Rs = Rvir / c

    mcum = consts * (np.log((Rs + r)/ Rs) + (Rs / (Rs + r))-1)
    
    return mcum


def nfw_profile_g(r, Mvir, c):
    M = nfw_profile_mcum(r, Mvir, c)
    r = M * u.M_sun * C.G / (r * u.kpc)**2
    return r.to('m/s**2').value



def mw_profile_faerman_n(r):
    """using their power law approximations"""
    n0 = 1.3e-5
    r_CGM = 283
    alpha = 0.93
    return n0 * (r/r_CGM)**(-alpha)

def mw_profile_faerman_T(r):
    """using their power law approximations"""
    T0 = 2.7e5
    r_CGM = 283
    alpha = 0.62
    return T0 * (r/r_CGM)**(-alpha)


def mw_profile_salem_n(r):
    n0 = 0.46
    rc = 0.35
    beta = 0.559
    return n0 * (1+(r/rc)**2)**(-3 * beta / 2.)

def mw_profile_salem_T(r):
    mu = 0.6
    gamma = 1.5
    Min = nfw_profile_mcum(r,1e12,c=12) * C.M_sun

    r = gamma * C.G * mu * C.m_p * Min / (3 * r * u.kpc * C.k_B)

    return r.to('K').value


def mw_profile_g(r):
    return nfw_profile_g(r, 1e12, 12)


def halo_profile_density(r):
    """
    y   : vertical coordinate (kpc or same length unit as H)
    rho0: midplane density
    a   : shape parameter
    H   : scale height
    """
    H_density = H
    Sigma_SI = (surface_density_Msun_pc2 * C.M_sun.to('g').value / (u.pc.to('cm')**2))
    rho0 = Sigma_SI / (2 * a_over_H * H_density)
    y_norm = r / (a_over_H * H_density)
    return rho0 * np.exp(-a_over_H * (np.sqrt(1 + y_norm**2) - 1))

def halo_profile_temperature(r):
    """
    Fixed-temperature profile (isothermal).
    """
    return 1e6


def halo_profile_g(r):
    """
    Compute vertical gravity g(y) in SI units.

    Parameters
    ----------
    y : float or array
        Vertical coordinate (meters)
    surface_density_Msun_pc2 : float
        Surface density Σ in solar masses / pc^2
    a_over_H : float
        Ratio a/H used in your profile
    H : float
        Scale height (meters)

    Returns
    -------
    g : float or array
        Gravitational acceleration at height y (m/s^2)
    """

    # Convert surface density from M_sun/pc^2 → kg/m^2
    Sigma_SI = surface_density_Msun_pc2 * C.M_sun.to('g') / (u.pc.to('cm')**2)
    H_density = H
    # Dimensionless vertical coordinate
    y_norm = r / (a_over_H * H_density)

    # Gravity formula
    g = 2 * np.pi * C.G.to(u.cm**3 / (u.g * u.s**2)) * Sigma_SI * y_norm / np.sqrt(1 + y_norm**2)

    return g.value