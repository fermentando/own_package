import os
import glob
import re
import numpy as np
from utils import *
from adjust_ics import *
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib import cm
from matplotlib.colors import Normalize
from cooling import get_c_s
from stratified_box import StratifiedBox
from turbulent_box import TurbulentBox
from cooling import get_t_cool_cgs

def read_hst(run): 
    data = np.loadtxt(os.path.join(run, 'out/parthenon.out1.hst'))
    data = np.where(data == 0, 1e-22, data)
    return data

def mass_evolution(run, gout=False):
    data = read_hst(run)
    mass_ind = 10
    norm_mass = np.log10(data[:, mass_ind] / data[0, mass_ind])
    timeseries = data[:, 0]
    wgout = np.zeros_like(timeseries)
    cgout = np.zeros_like(timeseries)
    total = norm_mass
    if gout: 
        wgout = np.log10(data[:, -2] / data[0, mass_ind]) 
        cgout = np.log10(data[:, -3] / data[0, mass_ind])
        total = np.log10((data[:, mass_ind] + data[:, -2] + data[:, -3]) / data[0, mass_ind])
    return timeseries, norm_mass, cgout, wgout, total

def vel_evolution(run):
    data = read_hst(run)
    vel_ind = 13
    mass_ind = 10
    timeseries = data[:, 0]
    velocity = (data[:, vel_ind] / data[:, mass_ind])
    return timeseries, velocity

def extract_r_value(path):
    """Extract the numeric value following 'r' in the path (cloud radius in parsecs)"""
    match = re.search(r'r([\d.]+)', path)
    if match:
        return float(match.group(1))
    return None

def get_mach_at_time(run_dir, target_teddy=1.2):
    """
    Get the Mach number at a specific time in units of eddy turnover time.
    Similar to the reference script.
    """
    data = np.loadtxt(os.path.join(run_dir, "out/parthenon.out1.hst"))
    
    try:
        run = StratifiedBox(os.path.join(run_dir, "strat.in"), run_dir)
        T0 = float(run.reader.get('problem/stratified_box', 'T_base'))
    except:
        run = TurbulentBox(os.path.join(run_dir, "turbulence.in"), run_dir)
        rho0 = float(run.reader.get('problem/turbulence', 'rho0')) * run.code_mass_cgs / run.code_length_cgs**3
        p0 = float(run.reader.get('problem/turbulence', 'p0')) * run.code_mass_cgs / run.code_length_cgs / run.code_time_cgs**2
        T0 = p0 / (rho0 * constants.kb / run.mbar)
    
    mach = float(run.reader.get('problem/turbulence', 'Mach_drive'))
    k_peak = float(run.reader.get('problem/turbulence', 'kpeak'))
    
    Lymin, Lymax = float(run.reader.get('parthenon/mesh', 'x2min')), float(run.reader.get('parthenon/mesh', 'x2max'))
    Lxmin, Lxmax = float(run.reader.get('parthenon/mesh', 'x1min')), float(run.reader.get('parthenon/mesh', 'x1max'))
    Lzmin, Lzmax = float(run.reader.get('parthenon/mesh', 'x3min')), float(run.reader.get('parthenon/mesh', 'x3max'))
    
    L_drive = Lxmax - Lxmin
    V_box = (Lxmax - Lxmin) * (Lymax - Lymin) * (Lzmax - Lzmin)
    
    cs = get_c_s(T0)
    velocity_cgs = run.code_length_cgs / run.code_time_cgs
    v_turb = cs * mach / velocity_cgs
    t_eddy = L_drive / v_turb
    
    t = data[:, 0]
    output_mach = data[:, -1]
    
    # Find the value at target_teddy
    t_normalized = t / t_eddy
    target_idx = np.argmin(np.abs(t_normalized - target_teddy))
    
    mach_value = output_mach[target_idx] / V_box
    
    return mach_value, t_eddy

if __name__ == "__main__":
    plt.style.use('custom_plot')
    
    # Find all matching runs
    base_pattern = '/viper/ptmp/ferhi/StratDisk/m0.*/r10/burnin'
    noturb_path = '/viper/ptmp/ferhi/StratDisk/noturb/r10'
    run_paths = sorted(glob.glob(base_pattern)) + glob.glob(noturb_path)
    
    if not run_paths:
        print(f"No runs found matching pattern: {base_pattern}")
        exit(1)
    
    print(f"Found {len(run_paths)} runs:")
    for run in run_paths:
        print(f"  {run}")
    
    # Extract Mach numbers and r values
    mach_values = []
    r_values = []
    valid_runs = []
    
    for run in run_paths:
        try:
            r_val = extract_r_value(run)
            if r_val is None:
                print(f"  Could not extract r value from {run}")
                continue
            
            mach_val, _ = get_mach_at_time(run, target_teddy=1.2)
            if 'noturb' in run: mach_val = 0
            
            mach_values.append(mach_val)
            r_values.append(r_val)
            valid_runs.append(run)
            print(f"  Extracted: Mach={mach_val:.4f}, r={r_val:.1f} pc")
            
        except Exception as e:
            print(f"  Error extracting values from {run}: {e}")
            continue
    
    if not valid_runs:
        print("No valid runs with extractable Mach and r values found!")
        exit(1)
    
    # Set up colormap
    norm = Normalize(vmin=min(mach_values), vmax=max(mach_values))
    cmap = cm.get_cmap('viridis')  # You can change to 'plasma', 'coolwarm', etc.
    
    # Create figure with four subplots
    fig, (ax_mass, ax_vel, ax_pos, ax_tgrow) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # Plot each run
    for run, mach_val, r_val in zip(valid_runs, mach_values, r_values):
        try:
            sim = StratifiedBox(os.path.join(run, 'strat.in'), dir=run)
            code_time_cgs = float(sim.reader.get('units', 'code_time_cgs'))
            code_length_cgs = float(sim.reader.get('units', 'code_length_cgs'))
            
            # Get mass and velocity evolution
            times, norm_mass, cgout, wgout, total = mass_evolution(run, gout=True)
            tvs, vel = vel_evolution(run)
            
            # Find injection time (similar to original script)
            try:
                mask = 10**norm_mass > 1
                idx0 = 0
            except:
                mask = 10**norm_mass > 0   
                idx0 = 0
            
            timeseries = times[idx0:] - times[idx0]
            norm_mass = norm_mass[idx0:] - norm_mass[idx0]
            vel = vel[idx0:]
            
            # Convert to physical units
            time_myr = timeseries * code_time_cgs / u.Myr.to('s')
            vel_kms = vel * code_length_cgs / code_time_cgs / 1e5
            
            # Remove artifact at beginning where velocity is -1 km/s
            # Find first point where velocity is significantly different from -1
            valid_mask = np.abs(vel_kms - 1.0) > 0.1  # velocity != -1 (with tolerance)
            if np.any(valid_mask):
                first_valid_idx = np.where(valid_mask)[0][0]
                time_myr = time_myr[first_valid_idx:]
                vel_kms = vel_kms[first_valid_idx:]
                norm_mass = norm_mass[first_valid_idx:]
                norm_mass = norm_mass - norm_mass[0]
                # Shift time so first valid point is at t=0
                time_myr = time_myr - time_myr[0]
            
            # Calculate y position: integrate velocity over time
            # Convert time from Myr to seconds for integration
            time_s = time_myr * u.Myr.to('s')
            vel_cm_s = vel_kms * 1e5  # km/s to cm/s
            
            # Integrate velocity to get position (using cumulative trapezoidal integration)
            y_position_cm = np.zeros_like(vel_cm_s)
            for i in range(1, len(vel_cm_s)):
                dt = time_s[i] - time_s[i-1]
                y_position_cm[i] = y_position_cm[i-1] + vel_cm_s[i] * dt
            
            # Convert position from cm to parsecs
            y_position_pc = y_position_cm / u.pc.to('cm')
            
            # Normalize by cloud radius
            y_position_normalized = y_position_pc / r_val
            
            # Calculate growth timescale: m / (dm/dt)
            # m is 10^(norm_mass) (in units of m0)
            # dm/dt is the derivative
            mass_linear = 10**norm_mass
            
            # Calculate dm/dt using np.gradient (handles edges automatically)
            dmdt = np.gradient(mass_linear, time_myr)
            
            # Growth timescale in Myr
            # Avoid division by zero or very small values
            growth_timescale = np.zeros_like(mass_linear)
            valid_growth = np.abs(dmdt) > 1e-10  # Only where growth rate is significant
            growth_timescale[valid_growth] = mass_linear[valid_growth] / dmdt[valid_growth]
            growth_timescale[~valid_growth] = np.nan  # Set to NaN where growth is negligible
            growth_timescale = np.convolve(growth_timescale, np.ones(1)/1, mode='same')
            
            # Get color from colormap
            color = cmap(norm(mach_val))
            
            # Plot mass evolution
            ax_mass.plot(time_myr, norm_mass, 
                        color=color, 
                        label=f'Mach={mach_val:.3f}', 
                        alpha=0.8,
                        linewidth=2)
            
            # Plot velocity evolution
            ax_vel.plot(time_myr, vel_kms, 
                       color=color, 
                       label=f'Mach={mach_val:.3f}', 
                       alpha=0.8,
                       linewidth=2)
            
            # Plot normalized position evolution
            ax_pos.plot(time_myr, y_position_normalized, 
                       color=color, 
                       label=f'Mach={mach_val:.3f}', 
                       alpha=0.8,
                       linewidth=2)
            
            # Plot growth timescale
            ax_tgrow.plot(time_myr, growth_timescale, 
                         color=color, 
                         label=f'Mach={mach_val:.3f}', 
                         alpha=0.8,
                         linewidth=2)
            
            print(f"Successfully plotted: {run} (Mach={mach_val:.4f}, r={r_val:.1f} pc)")

            Lambda_units = float(sim.reader.get('cooling', 'lambda_units_cgs'))
            #tgrow = sim.chi * np.sqrt(r_val * u.pc.to('m') / (150 * 1e3 * mach_val) * get_t_cool_cgs(sim.cloud_rho, sim.T_cloud, sim.mbar)*Lambda_units) / u.Myr.to('s')
            #print("this is tgrow: ", tgrow)
            #idx = np.argmin(abs(10 *tgrow - time_myr))
            #print("this is second time in myr: ", time_myr[1])
            #print("tgrow/tend:", idx/len(time_myr))
            #print("This is y at tgrow for sim:", y_position_normalized[idx])
            
        except Exception as e:
            print(f"Error processing {run}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Labels and formatting for mass plot
    ax_mass.set_ylabel(r'$\log(m/m_0)$', fontsize=14)
    ax_mass.set_ylim(bottom=-1, top=2)
    ax_mass.set_xlim(left=0)
    ax_mass.grid(True, alpha=0.3)
    ax_mass.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Labels and formatting for velocity plot
    ax_vel.set_ylabel(r'Speed [km/s]', fontsize=14)
    ax_vel.set_xlim(left=0)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Labels and formatting for position plot
    ax_pos.set_ylabel(r'$z/r_{\rm cloud}$', fontsize=14)
    ax_pos.set_xlim(left=0)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Labels and formatting for growth timescale plot
    ax_tgrow.set_xlabel(r't [Myr]', fontsize=14)
    ax_tgrow.set_ylabel(r'$t_{\rm grow} = m/\dot{m}$ [Myr]', fontsize=14)
    ax_tgrow.set_xlim(left=0)
    ax_tgrow.set_yscale('log')  # Log scale often useful for timescales
    ax_tgrow.grid(True, alpha=0.3)
    ax_tgrow.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax_mass, ax_vel, ax_pos, ax_tgrow], label='Mach number (at 1.2 $t_{eddy}$)', pad=0.02)

    
    # Save figure
    output_file = f'mass_velocity_position_evolution_all_runs_{r_val}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
