from utils import *
import utils as ut
from adjust_ics import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from gen_strat import *
from tqdm import tqdm
from scipy.integrate import RK45
from numba import njit



@njit
def rhs_numba(t, y, rho0, a, H, C0, A_cross, g, tcool_Myr, r_cl_pc, chi, s_to_Myrs):
    """
    Fast RHS for the ODE, compiled with Numba.
    """
    z, v, m = y

    # --- Inline t_grow_sub (avoid Python calls) ---
    v_safe = abs(v) if abs(v) > 1e-8 else 1e-8
    v_kms = v_safe / 1e5
    scaling_v = (150.0 / v_kms)**(3.0 / 5.0)
    scaling_tcool = (tcool_Myr / 0.03)**(1.0 / 4.0)
    scaling_r = (r_cl_pc / 100.0)**(3.0 / 4.0)
    scaling_chi = (chi / 100.0)
    t_grow = 35.0 * scaling_v * scaling_tcool * scaling_r * scaling_chi / s_to_Myrs

    # --- Density profile rho_hot(z) ---
    rho_local = rho0 * np.exp(-a * (np.sqrt(1.0 + (z / (a * H))**2) - 1.0))

    # --- Dynamics ---
    dm_dt = m / t_grow
    dv_dt = (m * g - 0.5 * rho_local * C0 * A_cross * v * abs(v) - v * dm_dt) / m
    dz_dt = v

    return np.array([dz_dt, dv_dt, dm_dt])


def velocity_fits(sim, rho_cloud_0, chi, rho_0, y_center, H):
    """
    Fast version of velocity_fits with Numba acceleration and improved solver settings.
    """
    # --- Coerce inputs to scalars ---
    rho_cloud_0 = float(np.asarray(rho_cloud_0).ravel()[0])
    rho_0 = float(np.asarray(rho_0).ravel()[0])
    y_center = float(np.asarray(y_center).ravel()[0])
    H = float(np.asarray(H).ravel()[0])
    chi = float(np.asarray(chi).ravel()[0])

    # --- Cloud properties ---
    r_cl_cm = float(sim.cloud_inserted * ut.constants.pc_to_cm)
    cshot = get_c_s(sim.T_base)
    t_cc = np.sqrt(chi) * r_cl_cm / 0.7 / cshot
    m0 = (4.0 / 3.0) * np.pi * rho_cloud_0 * r_cl_cm**3

    # --- Gravity ---
    g0_cgs = 2.0 * np.pi * ut.constants.G * sim.surface_density * sim.code_mass_cgs / sim.code_length_cgs**2

    # --- Cooling time ---
    tcool_s = get_t_cool_cgs(sim.T_base / chi, rho_cloud_0, sim.mbar)
    tcool_Myr = tcool_s * ut.constants.s_to_Myrs

    # --- Parameters (all in cgs) ---
    A_cross = np.pi * r_cl_cm**2
    C0 = 0.47
    rho0 = rho_0
    g = g0_cgs
    v0 = 0.0
    z0 = y_center
    a = float(sim.a_over_H)
    r_cl_pc = float(sim.cloud_inserted)
    s_to_Myrs = ut.constants.s_to_Myrs

    t_max = 20.0 * t_cc
    y0 = [z0, v0, m0]

    print(f"Integrating from t=0 to t={t_max:.2e} s (t_cc = {t_cc:.2e} s)")

    # --- Define RHS for solver (wraps numba function) ---
    def rhs(t, y):
        return rhs_numba(t, y, rho0, a, H, C0, A_cross, g, tcool_Myr, r_cl_pc, chi, s_to_Myrs)

    # --- Solve ODE ---
    sol = solve_ivp(
        rhs,
        (0, t_max),
        y0,
        method='LSODA',   # auto-switch stiff/non-stiff
        rtol=1e-3,
        atol=1e-4
    )

    print(f"Integration complete! Status: {sol.message}")

    t_vals = sol.t
    z_vals, v_vals, m_vals = sol.y

    # --- Plot results ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [cm/s]", color=color1)
    ax1.plot(t_vals, v_vals, color=color1, label="v(t)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Mass [g]", color=color2)
    ax2.plot(t_vals, m_vals, color=color2, linestyle="--", label="m(t)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.legend(loc="upper right")

    plt.title("Mass and Velocity Evolution with Stratified Density ρ(z)")
    plt.tight_layout()
    plt.savefig("mass_evol_fast.png")
    plt.close(fig)

    # z(t)
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, z_vals, color="tab:green")
    plt.xlabel("Time [s]")
    plt.ylabel("Height [cm]")
    plt.title("z(t)")
    plt.tight_layout()
    plt.savefig("z_evolution_fast.png")
    plt.close()

    # Diagnostics
    sample_v = 10.0e5
    v_safe = max(abs(sample_v), 1e-8)
    v_kms = v_safe / 1e5
    scaling_v = (150.0 / v_kms)**(3.0 / 5.0)
    scaling_tcool = (tcool_Myr / 0.03)**(1.0 / 4.0)
    scaling_r = (r_cl_pc / 100.0)**(3.0 / 4.0)
    scaling_chi = (chi / 100.0)
    sample_tgrow = 35.0 * scaling_v * scaling_tcool * scaling_r * scaling_chi / s_to_Myrs

    print("sample t_grow (s) at v=10 km/s:", sample_tgrow)

    return t_vals, z_vals, v_vals, m_vals

if __name__ == "__main__":
    localDir = os.getcwd()
    filename_input = os.path.join(localDir, 'restrat.in')
    sim = StratifiedBox(filename_input, os.path.abspath(os.path.join(filename_input, '..')))

    params = load_params(filename_input)
    code_mass_cgs = sim.code_mass_cgs
    code_length_cgs = sim.code_length_cgs
    nx1, nx2, nx3 = int(params['nx1']), int(params['nx2']), int(params['nx3'])
    mbar_over_kb = sim.mbar/ut.constants.kb 

    g0, H, rho_0 = gen_init_params(params, code_mass_cgs, code_length_cgs, mbar_over_kb)
    full_box_rho = isothermal_strat(nx1, nx2, nx3, rho_0, params['a_over_H'], H,
                (params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                (params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                (params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs),
                )
    rho_cloud_0, rho_hot_0, y_center = insert_sphere(full_box_rho, np.zeros_like(full_box_rho), np.zeros_like(full_box_rho), r=params['r_cloud_inserted'] * code_length_cgs, 
                            T_cloud=params['T_cloud'], 
                            mbar_over_kb=mbar_over_kb, gamma=params['gamma'], T_base = params['T_base'],
                            x_range=(params['x1min'] * code_length_cgs, params['x1max'] * code_length_cgs),
                            y_range=(params['x2min'] * code_length_cgs, params['x2max'] * code_length_cgs),
                            z_range=(params['x3min'] * code_length_cgs, params['x3max'] * code_length_cgs),
                            inplace=False, return_cloud_rho=True)
    
    chi = rho_cloud_0 / rho_hot_0


    velocity_fits(sim, rho_cloud_0, chi, rho_0, y_center, H)