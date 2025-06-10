import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from adjust_ics import SingleCloudCC

cm1 = sns.light_palette("seagreen", as_cmap=True)
cm2 = sns.color_palette("light:b", as_cmap=True)
cm3 = sns.light_palette("orange", as_cmap=True)

norm = mcolors.Normalize(vmin=0, vmax=5)
sm1 = ScalarMappable(cmap=cm1, norm=norm)
sm2 = ScalarMappable(cmap=cm2, norm=norm)
sm3 = ScalarMappable(cmap=cm3, norm=norm)

sm = [sm1, sm2, sm3]

def plot_vsf_subplots(npz_paths, outdir, stand_l=1.0, min_pairs=10):
    
    
    plt.style.use('custom_plot')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for subplot_idx in range(3):
        ax = axs[subplot_idx]
        for i in range(5):
            run_idx = subplot_idx * 5 + i
            print(run_idx)
            
            path = npz_paths[run_idx]
            sim = SingleCloudCC(os.path.join(path.split("out")[0], 'ism.in'), dir=os.path.dirname(path.split("out")[0]))
            print(path)

            # Load the saved VSF data
            data = np.load(path)
            vsf = data['vsf']
            log_centers = data['log_centers']

            # Apply mask
            vsf = np.where(np.isfinite(vsf) & (vsf >= 0), vsf, np.nan)

            # X-axis: separation distances
            correction_centers = 1
            if subplot_idx == 2: correction_centers = 10
            x_vals = 10 ** log_centers / stand_l * correction_centers
            lower_power = 10 ** np.floor(np.log10(min(x_vals)))

            ax.plot(x_vals, vsf/sim.v_wind, color = sm[subplot_idx].to_rgba(i+1))
            ax.set_xlim(left = lower_power)
            if i == 0: ax.vlines(x = 3 * correction_centers, ymin = 1e-1, ymax = 1e3, color='k', linestyle='--', linewidth=1, label=r'$r_\mathrm{crit}$', alpha= 0.4)
            if run_idx == 0: ax.legend(loc='upper left', fontsize = 12)

        # Reference slope line ~ l^{1/3}
        x_ref = x_vals[~np.isnan(vsf)]
        x0 = np.min(x_ref) * 1.5
        x1 = np.max(x_ref) / 4
        x_slope = np.array([x0, x1])
        y_slope = x_slope**(1/3)

        scale_factor = np.nanmax(vsf) / np.max(y_slope)
        y_slope *= scale_factor

        #ax.plot(x_slope, y_slope, 'k--', linewidth=1, label=r'$\propto l^{1/3}_{3D}$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel(r'$l_{3D} = \left| r_i - r_j \right| [r_{cl}]$', labelpad = 10, fontsize = 16)
        if subplot_idx == 0:
            ax.set_ylabel(r'$S_\mathrm{1,turb} = \langle | v_\mathrm{turb,i} - v_\mathrm{turb},j} | \rangle$', labelpad = 8, fontsize = 16)
            
            x0, x1 = 2*lower_power, 4*lower_power  # Choose x-range for reference line
            y0 = 1            # Starting y value

            # Draw the line: y = y0 * (x/x0)^-1
            x_vals = np.array([x0, x1])
            y_vals = y0 * (x_vals / x0)**(1/3)
            
            x_m2 = np.array([x0, x1])
            y_m2 = y0 * (x_m2/x0)**1
            ax.plot(x_vals, y_vals, color='k', linewidth=1)
            ax.plot(x_m2, y_m2, color='k', linewidth=1)
            ax.text(x1 * 1.2, y_vals[1] * 1.1, r'$\ell^{1/3}$', color='k', fontsize = 14, verticalalignment='top')
            ax.text(x_m2[1] * 1.1, y_m2[1] * 1.2, r'$\ell^{1}$', color='k', fontsize = 14, verticalalignment='top')
            plt.setp(ax.get_yticklabels(),fontsize = 14)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_ylim(bottom=5e-1, top=2e2)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "vsf_3x3_subplots.png")
    plt.savefig(outfile, dpi=300)
    plt.close()


if __name__ == "__main__":

    u1 = '/u/ferhi/Figures/velocity_structure_function/fvLism/kc/fv01_shorter/'
    u2 = '/u/ferhi/Figures/velocity_structure_function/fvLism/01kc/fv01_30r/'
    u3 = '/u/ferhi/Figures/velocity_structure_function/fvLism/01kc/fv02/'

    
    vsf_files = [
        u1+"3d_vsf_011.npz", u1+"3d_vsf_012.npz", u1+"3d_vsf_013.npz", u1+"3d_vsf_014.npz", u1+"3d_vsf_022.npz",
        u2+"3d_vsf_006.npz", u2+"3d_vsf_007.npz", u2+"3d_vsf_008.npz", u2+"3d_vsf_009.npz", u2+"3d_vsf_010.npz",   
        u3+"3d_vsf_004.npz", u3+"3d_vsf_006.npz", u3+"3d_vsf_020.npz", u3+"3d_vsf_026.npz", u3+"3d_vsf_037.npz",
        
    ]
    
        

    plot_vsf_subplots(np.sort(vsf_files), outdir='/u/ferhi/Figures/')