"""
Main script for adjusting AthenaK input files and initial conditions.

This script provides a command-line interface for adjusting cloud simulations,
including single cloud-wind interactions, stratified disks, and turbulent boxes.

Usage:
    Single Cloud (ism.in):
        python adjust_ics.py check
        python adjust_ics.py adjust <ratio>
        python adjust_ics.py enlarge_y [factor]
        python adjust_ics.py enlarge_x [factor]
        python adjust_ics.py res [factor]
        python adjust_ics.py mach_shock
    
    Stratified Box (strat.in):
        python adjust_ics.py check
        python adjust_ics.py enlarge_y [factor]
        python adjust_ics.py enlarge_x [factor]
        python adjust_ics.py rescale [factor]
"""

import os
import sys
from single_cloud import SingleCloudCC
from stratified_box import StratifiedBox
from turbulent_box import TurbulentBox


def main():
    """Main entry point for the script."""
    localDir = os.getcwd()
    
    # Check which type of simulation this is
    if os.path.isfile(os.path.join(localDir, "ism.in")):
        sim = SingleCloudCC(os.path.join(localDir, 'ism.in'), dir=localDir)
        _handle_single_cloud_command(sim)
        
    elif os.path.isfile(os.path.join(localDir, "strat.in")):
        sim = StratifiedBox(os.path.join(localDir, 'strat.in'), dir=localDir)
        _handle_stratified_command(sim)

    elif os.path.isfile(os.path.join(localDir, "turbulence.in")):
        sim = TurbulentBox(os.path.join(localDir, 'turbulence.in'), dir=localDir)
        _handle_turbulence_command(sim)
    
    else:
        print("Error: Neither 'ism.in' nor 'strat.in' found in current directory.")
        sys.exit(1)


def _handle_single_cloud_command(sim):
    """Handle commands for single cloud simulations."""
    if len(sys.argv) < 2:
        print("Error: No command provided. See help documentation.")
        sys.exit(1)
    
    command = str.lower(sys.argv[1])
    
    if command == "check":
        sim._modify_shock_mach()
        sim.state_ICs()
    elif command == "adjust":
        if len(sys.argv) < 3:
            print("Error: 'adjust' requires a ratio argument.")
            sys.exit(1)
        sim.reset_survival(float(sys.argv[2]), 8)
    elif command == "enlarge_y":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        sim.enlarge_dim(increase_factor=factor, axs=[2])
    elif command == "enlarge_x":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        sim.enlarge_dim(increase_factor=factor, axs=[1, 3])
    elif command == "res":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 8
        sim.set_rin_res(resol_factor=factor)
    elif command == "mach_shock":
        sim._modify_shock_mach()
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: check, adjust, enlarge_y, enlarge_x, res, mach_shock")
        sys.exit(1)


def _handle_stratified_command(sim):
    """Handle commands for stratified box simulations."""
    if len(sys.argv) < 2:
        print("Error: No command provided. See help documentation.")
        sys.exit(1)
    
    command = str.lower(sys.argv[1])
    
    if command == "check":
        sim._enforce_cartesian_grid()
        sim.stateICs()
    elif command == "enlarge_y":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        sim.enlarge_dim(increase_factor=factor, axs=[2])
    elif command == "enlarge_x":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        sim.enlarge_dim(increase_factor=factor, axs=[1, 3])
    elif command == "rescale":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 8
        sim.set_rin_res(resol_factor=factor)
    elif command == "cooling":
        sim.compute_restart_cooling_time()
    elif command == "set_y":
        sim.set_y(float(sys.argv[2])*1000)
    elif command == "radius":
        sim.radius(float(sys.argv[2]))
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: check, enlarge_y, enlarge_x, rescale")
        sys.exit(1)

def _handle_turbulence_command(sim):
    """Handle commands for stratified box simulations."""
    if len(sys.argv) < 2:
        print("Error: No command provided. See help documentation.")
        sys.exit(1)
    
    command = str.lower(sys.argv[1])
    
    if command == "check":
        sim._enforce_cartesian_grid()
        sim.state_ICs()
    elif command == "enlarge_y":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        sim.enlarge_dim(increase_factor=factor, axs=[2])
    elif command == "enlarge_x":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        sim.enlarge_dim(increase_factor=factor, axs=[1, 3])
    elif command == "rescale":
        factor = float(sys.argv[2]) if len(sys.argv) == 3 else 8
        sim.set_rin_res(resol_factor=factor)
    elif command == "cooling":
        sim.compute_restart_cooling_time()
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: check, enlarge_y, enlarge_x, rescale")
        sys.exit(1)


if __name__ == "__main__":
    main()
