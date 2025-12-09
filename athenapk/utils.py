import matplotlib.pyplot as plt 
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse
import matplotlib.pyplot as plt
import matplotlib.style as style
import yt
from yt.units import dyn, cm, K
import os

style.core.USER_LIBRARY_PATHS.append('custom_plot')

### Essentials ###
homeDir = '/u/ferhi'


@yt.derived_field(name=("gas", "pressure_normalized"), units="", sampling_type="cell")
def _pressure_normalized(field, data):
    p = data[("gas", "pressure")]
    pmax = 1.5e-11 * (dyn / cm**2)
    return p / pmax if pmax > 0 else p

@yt.derived_field(name=("gas", "mixing_gas_flag"), units="", sampling_type="cell")
def _temperature_range_flag(field, data):
    T = data[("gas", "temperature")]
    mask = (T >= 7e4 * yt.units.K) & (T <= 3e5 * yt.units.K)
    return mask.astype("float")

@yt.derived_field(name=("gas", "luminosity_normalized"), units="", sampling_type="cell")
def _luminosity_normalized(field, data):
    # Temperature field
    T = data[("gas", "temperature")]
    T_min = 0.8e5 * K
    T_max = 1.2e5 * K
    mask = (T >= T_min) & (T <= T_max)

    vx = data[("gas", "velocity_x")]
    vy = data[("gas", "velocity_y")]
    vz = data[("gas", "velocity_z")]
    v_mag = (vx**2 + vy**2 + vz**2)**0.5


    cs = data[("gas", "sound_speed")]
    M = v_mag / cs

    L = T * (1 + M**2) * mask

    # Normalize by max (avoid division by zero)
    Lmax = L.max()
    if Lmax > 0:
        L = L / Lmax
    return (L / Lmax).d if Lmax > 0 else L.d
fields = {

    'density': ("gas", "density"),
    #'pressure': ("gas", "pressure_normalized"),
    'temperature': ("gas", "temperature"),
    'velocity_y': ("gas", "velocity_y"),
    #'velocity_z': ('gas', 'velocity_z'),
    #'scalar':   ("gas", "mixing_gas_flag"),

}

### Constants ###
class constants:
    mh = 1.660538921e-24
    uam = 1.007947 *mh #cgs
    kb = 1.3806488e-16 #cgs
    kpc_over_cm = 3.24078e-22
    s_to_Myrs = 3.1710e-14 
    pc_to_cm = 3.086e+18
    G = 6.67430e-8 #cgs

    Xsol = 1.0
    Zsol = 1.0

    X = Xsol * 0.7381
    Z = Zsol * 0.0134
    Y = 1 - X - Z
    mu = 1.0 / (2.0 * X + 3.0 * (1.0 - X - Z) / 4.0 + Z / 2.0)      

    def __init__(self):
        raise TypeError("This class is a constants container and cannot be instantiated.")


### Classes ###
class ParameterNotFoundError(Exception):
    def __init__(self, section, parameter):
        self.section = section
        self.parameter = parameter
        super().__init__(f"Parameter '{parameter}' not found in section '{section}'")


class AthenaPKInputFileReader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.params = {}
        with open(file_name, 'r') as f:
            section = None
            for line in f:
                line = line.split('#')[0].strip()
                if line.startswith("<") and line.endswith(">"):
                    section = line[1:-1]
                    self.params[section] = {}
                elif "=" in line:
                    key, value = map(str.strip, line.split('=', 1))
                    if section:
                        self.params[section][key] = value

    def get(self, section, parameter, default=None, raise_error=False):
        param = self.params.get(section, {}).get(parameter, default)
        if param == None: raise ParameterNotFoundError(section, parameter)
        else: return param


    def set_(self, section, parameter, value):
        """Set or update a parameter value."""
        if section not in self.params:
            self.params[section] = {}
        self.params[section][parameter] = value

    def change_aspect_xlim(self, section, parameter, value):
        """Adjust limits of meshblock."""
        new_val = float(self.params[section][parameter] )
        self.params[section][parameter] =  new_val *float(value)


    def save(self):
        """Save the updated parameters back to the file."""
        with open(self.file_name, 'w') as f:
            for section, parameters in self.params.items():
                f.write(f"<{section}>\n")
                for key, value in parameters.items():
                    f.write(f"{key} = {value}\n")
                f.write("\n") 

### Parallel io ###
def process_file(args):
    run, output_dir, func = args
    func(run, output_dir)

def run_parallel(runs, func, num_workers, output_dir):
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, [(run, output_dir, func) for run in runs])
        
    results = [r for r in results if r is not None]

    if not results:
        return [] 
    return zip(*results)

def get_n_procs():
    parser = argparse.ArgumentParser(description="Set the number of processors.")
    parser.add_argument("--N_procs", nargs="?", type=int, default=1, help="Number of processors to use.")
    args = parser.parse_args()
    return max(1, min(args.N_procs, cpu_count()))

def get_user_args(sys_argvs):
    user_args = []
    skip_next = False

    for arg in sys_argvs[1:]:
        if skip_next:  
            skip_next = False  
            continue  
        if arg == "--N_procs":  
            skip_next = True  # Skip the next argument as well
            continue  
        user_args.append(arg)
    print("Arguments received:", sys_argvs)
    print("user args: ", user_args)
    
    return user_args

def get_n_procs_and_user_args():
    parser = argparse.ArgumentParser(description="Set the number of processors.")
    parser.add_argument("--N_procs", type=int, default=1, help="Number of processors to use.")
    
    args, remaining_args = parser.parse_known_args()
    n_procs = max(1, min(args.N_procs, cpu_count()))

    return n_procs, remaining_args

def get_working_dirs():

    N_procs, user_args = get_n_procs_and_user_args()
    print(f"N_procs set to: {N_procs} processors.")
    gout = True
    
    if len(user_args) > 0:
        RUNS = [os.getcwd()]
        run_paths = RUNS
        parts = RUNS[0].split('/')
        saveFile = f"{parts[-1]}"
        print('Saved to: ', saveFile)
        
    else:
        runDir = os.getcwd()
        run_paths = np.array([
            os.path.join(runDir, folder) 
            for folder in os.listdir(runDir) 
            if os.path.isdir(os.path.join(runDir, folder)) and 'strat.in' in os.listdir(os.path.join(runDir, folder)) 
        ])
        parts = runDir.split('/')
        saveFile = f"{parts[-2]}/{parts[-1]}"
        if not os.path.exists(os.path.join('/u/ferhi/Figures/',parts[-2])): 
            os.makedirs(os.path.join('/u/ferhi/Figures/',parts[-2]))

    return run_paths, saveFile