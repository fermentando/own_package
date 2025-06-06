CONST_pc = 3.086e18
CONST_yr = 3.154e7
CONST_amu = 1.66053886e-24
CONST_kB = 1.3806505e-16

unit_length = CONST_pc * 1e3  # 1 kpc
unit_time = CONST_yr * 1e6  # 1 Myr
unit_density = CONST_amu  # 1 mp/cm-3
unit_velocity = unit_length / unit_time

KELVIN = unit_velocity * unit_velocity * CONST_amu / CONST_kB
unit_q = (unit_density * (unit_velocity**3)) / unit_length

Xsol = 1.0
Zsol = 1.0

X = Xsol * 0.7381
Z = Zsol * 0.0134
Y = 1 - X - Z

mu = 1.0 / (2.0 * X + 3.0 * (1.0 - X - Z) / 4.0 + Z / 2.0)
mue = 2.0 / (1.0 + X)
muH = 1.0 / X

mH = 1.0

gamma = 5 / 3
g = 5 / 3


def B_mag_fn(Ma, M, T_hot, amb_rho):
    import numpy as np

    beta_arr = (2 / g) * (Ma / M) ** 2
    P_th = (T_hot / (KELVIN * mu)) * amb_rho
    P_B = P_th / beta_arr
    # Assuming that cgs relation between B_mag and P_B is used
    B_mag = np.sqrt(P_B * 2.0)

    return B_mag


def temperature(rho, prs):
    return (prs / rho) * (KELVIN * mu)


# TODO: Move all these into a class, whose constructor takes Xsol, Ysol values