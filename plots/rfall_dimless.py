import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from cooling import load_cooling_table, get_c_s
from stratified_box import StratifiedBox

ref_sim = StratifiedBox("/viper/ptmp/ferhi/StratDisk/Rfall/r10pc_t1e5_v3/strat.in", '.')

l_cool = np.array([6.5e-7, 1.874e-02, 3.860e-05, 2.7e-3, 1e-5, 1e-1, 1e-3, 1e-5])* u.Myr.to('s') * get_c_s(1e4) / u.pc.to('cm')#in pc
print(l_cool)
radius = np.array([6, 50, 0.1, 0.1, 1000, 4, 10 , 10])
status = ["yes", "no", "yes", "yes", "no", "no", "no", "yes"] #does it stays up?

x1 = np.logspace(-5, 1, 500)   # from 1e-6 to 1
y1 = 3e8 * x1

plt.style.use("custom_plot")

fig, ax = plt.subplots(figsize=(8, 6))

for y, x, s in zip(l_cool, radius, status):
    marker = "^" if s=="yes" else "v"
    color = "blue" if s=="yes" else "red"
    ax.scatter(x/4e3, 4e3/y, marker=marker, s=100, color=color)
ax.plot(x1, y1, "k--", label=r"$\propto \chi^{5/2} / \mathcal{M}^3 $")
#ax.plot(x1, y2, "g--", label=r"$t_{\rm cool} / t_{\rm eddy}$")
ax.vlines(1e-4, ymin=1e3, ymax=1e9, color = "black", linestyles="dashed", alpha = 0.5, label=r"$ =\mathcal{M}^2 / \chi $")
#ax.hlines(4e-1, xmin=1e-7, xmax=1, color = "black", linestyles="dashed", alpha = 0.5, label=r"$r_{\rm fall, g}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r'$ H / l_{\rm cool} $')
ax.set_xlabel(r"$r_{cl} / H$")
plt.legend()

plt.tight_layout()
plt.savefig("/u/ferhi/own_package/plots/infall_rcl.png", dpi=300)
