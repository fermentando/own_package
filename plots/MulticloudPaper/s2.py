import numpy as np
import matplotlib.pyplot as plt
from adjust_ics import *
from utils import *
import constants as ct
from scipy.optimize import curve_fit

def compute_structure_function(input_dir, p, max_lag):
    
    print('Computing structure function...')
    sim = SingleCloudCC(os.path.abspath(os.path.join(input_dir, 'ism.in')), input_dir)
    params = sim.reader
    unit_density= float(params.get('units', 'code_mass_cgs')) / float(params.get('units', 'code_length_cgs'))**3
    ICs, kval = sim._return_ICs()
    array = ICs[:,:,:,0]
    structure_function = np.zeros(max_lag)
    Nx, Ny, Nz = array.shape
    
    # Loop over the lags (1 to max_lag)
    for r in range(1, max_lag + 1):
        sum_diff = 0
        count = 0
        
        # Compute squared (S2) or cubed (S3) differences for valid indices
        if Nx > r:
            sum_diff += np.mean(np.abs(array[r:, :, :] - array[:-r, :, :]) ** p)
            count += 1
        if Ny > r:
            sum_diff += np.mean(np.abs(array[:, r:, :] - array[:, :-r, :]) ** p)
            count += 1
        if Nz > r:
            sum_diff += np.mean(np.abs(array[:, :, r:] - array[:, :, :-r]) ** p)
            count += 1
        
        # Store the mean of the computed differences
        if count > 0:
            structure_function[r - 1] = sum_diff / count
        print(max_lag - r)
    
    return structure_function / unit_density ** p



if __name__ == "__main__":
    input_dir = "/ptmp/ferhi/ISM_thinslab/kc/fv01_3.8rcl"
    
    max_lag = 100  # Set the maximum lag
    S2 = compute_structure_function(input_dir, p=2, max_lag=max_lag)  # Second-order structure function
    S3 = compute_structure_function(input_dir, p=3, max_lag=max_lag)  # Third-order structure function

    def model(S3, A, alpha):
        return A * S3**alpha


    params, covariance = curve_fit(model, S3, S2)
    A_fit, alpha_fit = params

    plt.figure(figsize=(8, 6))
    plt.scatter(S2, S3, color='b', label='S2 vs S3', alpha=0.7)
    plt.savefig('Zstructure.png')
    plt.show()


    S2_fitted = model(S3, A_fit, alpha_fit)

    plt.plot(S3, S2_fitted, color='r', label=f'Fit: S2 = {A_fit:.2f} * S3^{alpha_fit:.2f}')
    plt.text(0.1, 0.9, f'A = {A_fit:.2f}\nα = {alpha_fit:.2f}', transform=plt.gca().transAxes, fontsize=12, color='red')


    # Plot S2 against S3
    plt.figure(figsize=(8, 6))
    plt.scatter(S2, S3, color='b', label='S2 vs S3', alpha=0.7)
    plt.xlabel('S2 (second-order structure function)')
    plt.ylabel('S3 (third-order structure function)')
    plt.title('Plot of S2 against S3')
    plt.legend()
    plt.grid(True)
    plt.savefig('s2fit.png')
    plt.show()