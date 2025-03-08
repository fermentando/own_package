import matplotlib.pyplot as plt 
import numpy as np


homeDir = '/u/ferhi'

fields = {

    'density': ("gas", "density"),
    'pressure': ("gas", "pressure"),
    'velocity_y': ("gas", "velocity_y"),
    'temperature': ("gas", "temperature"),


}


class constants:
    mh = 1.660538921e-24
    uam = 1.007947 *mh #cgs
    kb = 1.3806488e-16 #cgs
    kpc_over_cm = 3.24078e-22
    s_to_Myrs = 3.1710e-14 

    Xsol = 1.0
    Zsol = 1.0

    X = Xsol * 0.7381
    Z = Zsol * 0.0134
    Y = 1 - X - Z
    mu = 1.0 / (2.0 * X + 3.0 * (1.0 - X - Z) / 4.0 + Z / 2.0)      

    def __init__(self):
        raise TypeError("This class is a constants container and cannot be instantiated.")



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


