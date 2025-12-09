import matplotlib.pyplot as plt
import numpy as np

cooling = [6.5e-7, 1.874e-02, 3.860e-05, 2.7e-3, 1e-5, 1e-1, 1e-3, 1e-5, 1e-4]
radius = [6, 50, 0.1, 0.1, 1000, 4, 10 , 10, 100]
status = ["yes", "no", "yes", "yes", "no", "no", "no", "yes", "no"] #does it stays up?

x1 = np.logspace(-6, 0, 500)   # from 1e-6 to 1
y2 = 10 * x1
y1 = 1e-3 / x1    

plt.style.use("custom_plot")

fig, ax = plt.subplots(figsize=(8, 6))

for x, y, s in zip(cooling, radius, status):
    marker = "^" if s=="yes" else "v"
    color = "blue" if s=="yes" else "red"
    ax.scatter([x], [y], marker=marker, s=100, color=color)
ax.plot(x1, y1, "k--", label=r"$r_{\rm fall, \dot{m}} \propto t_{\rm cool}^{-1}$")
ax.plot(x1, y2, "g--", label=r"$t_{\rm cool} / t_{\rm eddy}$")
ax.hlines(4e-1, xmin=1e-7, xmax=1, color = "black", linestyles="dashed", alpha = 0.5, label=r"$r_{\rm fall, g}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Cooling Timescale (Myr)")
ax.set_ylabel("Cloud Radius (pc)")
plt.legend()
plt.ylim(bottom = 1e-2)

plt.tight_layout()
plt.savefig("/u/ferhi/own_package/plots/cloud_radius_cooling.png", dpi=300)
