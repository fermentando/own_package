import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from utils import constants as const
import matplotlib.pyplot as plt

# Read cooling table
kb = const.kb
# Assuming the file has two columns: T (temperature), cooling_value
table_path = "/u/ferhi/athenapk-fork-fernando/inputs/cooling_tables/gnat-sternberg.cooling_1Z"
data = np.loadtxt(table_path)
T_table = pow(10, np.array(data[:, 0]))
cooling_table = pow(10, np.asarray(data[:, 1]))

# Interpolation function for cooling
cooling_interp = interp1d(T_table, cooling_table, kind='cubic', bounds_error=False, fill_value=(cooling_table[0], cooling_table[-1]))

# Compute a constant pressure value (example)
pressure_value = kb * 1* 1e4  # Example: n=1e4 cm^-3, T=1e6 K

# Function to minimize
def cooling_function(T, pressure_value):
    return kb**2 * T**2 / abs(pressure_value * cooling_interp(T))

# Function to find T that minimizes the cooling function
def tcoolmin(pressure_value, Tmin=1e4, Tmax=1e6):
    result = minimize_scalar(
        cooling_function, 
        bounds=(Tmin, Tmax), 
        args=(pressure_value,), 
        method='bounded'
    )
    if result.success:
        return result.x, result.fun
    else:
        raise RuntimeError("Minimization did not converge.")

# Example usage
T_min, f_min = tcoolmin(pressure_value)
print(f"Temperature at minimum: {T_min:.2e} K")
print(f"Minimum value of function in Myrs: {f_min/3600/24/365/1e6:.3e}")

T_plot = np.logspace(4, 6, 500)  # 1e4 to 1e6 K
f_plot = abs(cooling_interp(T_plot))


"""
plt.figure(figsize=(8,5))
plt.plot(T_plot, f_plot, label='Cooling Function')
plt.axvline(T_min, color='r', linestyle='--', label=f'Min at {T_min:.2f} K')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Temperature (K)')
plt.ylabel('Cooling Function f(T)')
plt.title('Cooling Function vs Temperature')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig("cooling_function.png", dpi=300)
"""