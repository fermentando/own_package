import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from adjust_ics import SingleCloudCC
from adjust_ics import get_c_s
import matplotlib as mpl
import matplotlib.gridspec as gridspec


cm1 = sns.light_palette("seagreen", as_cmap=True)
cm2 = sns.color_palette("light:b", as_cmap=True)
cm3 = sns.light_palette("orange", as_cmap=True)

norm = mcolors.Normalize(vmin=0, vmax=5)
sm1 = ScalarMappable(cmap=cm1, norm=norm)
sm2 = ScalarMappable(cmap=cm2, norm=norm)
sm3 = ScalarMappable(cmap=cm3, norm=norm)

sm = [sm1, sm2, sm3]

legends = [
    r'$(1, 10^{-1}, 30)$', 
    r'$(1, 10^{-2}, 300)$', 
    r'$(0.1, 10^{-1}, 300)$', 
]

def plot_vsf_subplots(npz_paths, outdir, stand_l=1.0, min_pairs=10):
    
    
    plt.style.use('custom_plot')
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.2)

    axs = [fig.add_subplot(gs[i]) for i in range(3)]  # subplots
    cbar_ax = fig.add_subplot(gs[3])   
    

    for subplot_idx in range(3):
        ax = axs[subplot_idx]
        for i in range(5):
            run_idx = subplot_idx * 5 + i
            print(run_idx)
            
            path = npz_paths[run_idx]
            input_file_path = '/viper/ptmp/ferhi/LEGACY/' + path.split("3d_vsf_")[0].split("function")[-1]
            sim = SingleCloudCC(os.path.join(input_file_path, 'ism.in'), dir=input_file_path)
            depth = float(sim.reader.get('problem/wtopenrun', 'depth'))
            l_turbulent =  depth 
            cs = get_c_s(sim.T_cloud)/1e5  # in km/s
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

            if i ==4 :ax.plot(x_vals, np.sqrt(3/2)*vsf/cs, color = sm[subplot_idx].to_rgba(i+1), label = legends[subplot_idx])
            else: ax.plot(x_vals, np.sqrt(3/2)*vsf/cs, color = sm[subplot_idx].to_rgba(i+1))
            ax.set_xlim(left = lower_power)
            if i == 0: ax.vlines(x = l_turbulent, ymin = 1e-2, ymax = 1e3, color='k', linestyle='--', linewidth=1, label=r'$L_\mathrm{ISM}$', alpha= 0.4)
            ax.legend(loc='upper left', fontsize = 12)

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
            ax.set_ylabel(r'$\bar{S}_\mathrm{1} = \langle | v(r + \ell) - v(r) | \rangle / c_\mathrm{s,cold}$', labelpad = 8, fontsize = 16)
            
            x0, x1 = 2*lower_power, 4*lower_power  # Choose x-range for reference line
            y0 = 1.5e-1            # Starting y value

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
        ax.set_ylim(bottom=1e-1, top=10)

    ## Colormap
    gray_cmap = mpl.colors.ListedColormap([str(0.2 + 0.15*j) for j in range(5)][::-1])  # 5 tones of gray
    bounds = np.arange(1, 7)  # 1 to 5 labels
    gray_norm = mpl.colors.BoundaryNorm(bounds, gray_cmap.N)
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=gray_cmap, norm=gray_norm, orientation='vertical'
    )
    cbar.set_ticks(np.arange(1.5, 6))
    cbar.set_ticklabels([f"{i:.2f}" for i in np.linspace(0.1,1,5)])
    cbar.set_label(r'$ t / t_\mathrm{sh}$', rotation=0, labelpad=15, size = 16)


    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "consistent_vsf_3x3_subplots.pdf")
    print(f"Saving to {outfile}")
    axs[1].set_title(r'$(r_\mathrm{cl}/r_\mathrm{crit}, f_v, L_\mathrm{ISM}/r_\mathrm{cl})$', fontsize=16, y=1.02)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    u1 = '/u/ferhi/Figures/velocity_structure_function/fvLism/kc/fv01_shorter/'
    u2 = '/u/ferhi/Figures/velocity_structure_function/fvLism/01kc/fv01_30r/'
    u3 = '/u/ferhi/Figures/velocity_structure_function/fvLism/01kc/fv02/'

    
    vsf_files = [
        u1+"3d_vsf_006.npz", u1+"3d_vsf_008.npz", u1+"3d_vsf_012.npz", u1+"3d_vsf_014.npz", u1+"3d_vsf_020.npz",
        u2+"3d_vsf_004.npz", u2+"3d_vsf_006.npz", u2+"3d_vsf_008.npz", u2+"3d_vsf_009.npz", u2+"3d_vsf_010.npz",   
        u3+"3d_vsf_004.npz", u3+"3d_vsf_006.npz", u3+"3d_vsf_008.npz", u3+"3d_vsf_013.npz", u3+"3d_vsf_014.npz",
        
    ]

        
    plot_vsf_subplots(np.sort(vsf_files), outdir='/u/ferhi/Figures/')