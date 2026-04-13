"""
Reproduce Fig. 1 of Chevalier & Clegg (1985):
log(u*), log(rho*), log(P*) vs log(r/R)

Dimensionless scalings:
  u*   = u   / (Mdot^{-1/2} Edot^{1/2})
  rho* = rho / (Mdot^{3/2} Edot^{-1/2} R^{-2})
  P*   = P   / (Mdot^{1/2} Edot^{1/2} R^{-2})
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

GAMMA = 5/3
MDOT  = 1.0
EDOT  = 1.0
R     = 1.0

# ── Implicit Mach-number equations ───────────────────────────────────────────

def lhs_inner(M, g=GAMMA):
    e1 = -(3*g + 1) / (5*g + 1)
    e2 =  (g + 1)   / (2*(5*g + 1))
    return ((3*g + 1/M**2) / (1 + 3*g))**e1 * \
           ((g - 1 + 2/M**2) / (1 + g))**e2

def lhs_outer(M, g=GAMMA):
    e = (g + 1) / (2*(g - 1))
    return M**(2/(g-1)) * ((g - 1 + 2/M**2) / (1 + g))**e

def solve_M_inner(xi):          # xi = r/R, subsonic branch
    if xi >= 1.0: return 1.0
    return brentq(lambda M: lhs_inner(M) - xi, 1e-7, 1-1e-10)

def solve_M_outer(xi):          # xi = r/R, supersonic branch
    if xi <= 1.0: return 1.0
    return brentq(lambda M: lhs_outer(M) - xi**2, 1+1e-10, 1e4)

# ── Reference state at sonic point r = R ─────────────────────────────────────

def ref_state(g=GAMMA, Mdot=MDOT, Edot=EDOT, Rv=R):
    u_R   = np.sqrt(2*(g-1)/(g+1) * Edot/Mdot)
    rho_R = Mdot / (4*np.pi * Rv**2 * u_R)
    P_R   = rho_R * u_R**2 / g
    return u_R, rho_R, P_R

# ── Physical u, rho, P from M (Bernoulli + adiabatic) ────────────────────────

def phys(M, u_R, rho_R, P_R, g=GAMMA):
    u_n   = M * np.sqrt((g+1) / (2 + (g-1)*M**2))   # u / u_R
    c_n   = u_n / M                                    # c / c_R
    rho_n = c_n**(2/(g-1))
    P_n   = c_n**(2*g/(g-1))
    return u_R*u_n, rho_R*rho_n, P_R*P_n

# ── Dimensionless scalings from the paper ────────────────────────────────────

def scale(u, rho, P, Mdot=MDOT, Edot=EDOT, Rv=R):
    u_star   = u   / (Mdot**(-0.5) * Edot**0.5)
    rho_star = rho / (Mdot**1.5 * Edot**(-0.5) * Rv**(-2))
    P_star   = P   / (Mdot**0.5 * Edot**0.5 * Rv**(-2))
    return u_star, rho_star, P_star

# ── Build radial grid ─────────────────────────────────────────────────────────

n = 400
# log-spaced from r/R = 10^{-0.55} to 10^{0.55}  (matches paper's x-axis range)
xi_all = np.logspace(-0.55, 0.55, n)

u_R, rho_R, P_R = ref_state()

log_xi  = []
log_us  = []
log_rs  = []
log_Ps  = []

for xi in xi_all:
    if xi < 1.0:
        M = solve_M_inner(xi)
    else:
        M = solve_M_outer(xi)
    u, rho, P = phys(M, u_R, rho_R, P_R)
    us, rs, Ps = scale(u, rho, P)
    log_xi.append(np.log10(xi))
    log_us.append(np.log10(us))
    log_rs.append(np.log10(rs))
    log_Ps.append(np.log10(Ps))

log_xi = np.array(log_xi)
log_us = np.array(log_us)
log_rs = np.array(log_rs)
log_Ps = np.array(log_Ps)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(log_xi, log_us, 'k-', lw=1.6)
ax.plot(log_xi, log_rs, 'k-', lw=1.6)
ax.plot(log_xi, log_Ps, 'k-', lw=1.6)

# Labels near the curves (matching paper positions)
ax.text(0.32,  0.05, r'$\log\,(u_*)$',   fontsize=12, ha='center')
ax.text(0.35, -1.35, r'$\log\,(\rho_*)$', fontsize=12, ha='center')
ax.text(0.20, -3.15, r'$\log\,(P_*)$',   fontsize=12, ha='center')

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-4.0, 1.1)
ax.set_xlabel(r'$\log\,(r/R)$', fontsize=13)
ax.set_ylabel('')
ax.set_yticks(np.arange(-4, 2, 1))
ax.set_yticklabels([str(int(v)) if v != 0 else '0.0' for v in np.arange(-4, 2, 1)])

# Minimal spine style to match the paper's clean look
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.tick_params(direction='in', which='both', top=True, right=True)
ax.set_yticks([-4,-3,-2,-1,0,1])
ax.set_yticklabels(['-4.0','-3.0','-2.0','-1.0','0.0','1.0'])
ax.axvline(0, color='k', lw=0.5, ls=':')

ax.set_title(r'Fig. 1 reproduction — $\gamma = 5/3$', fontsize=11, pad=8)

plt.tight_layout()
plt.savefig('/u/ferhi/Figures/CC85.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("Figure saved to /u/ferhi/Figures/CC85.png")