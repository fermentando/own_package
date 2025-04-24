#!/bin/sh -l
# Standard output and error:
#SBATCH -o ./proj.%j.log
#SBATCH -e ./proj.%j.log
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J 2quick_projections_gpu

# Request 1 GPU (APU if needed)
#SBATCH --gres=gpu:2
#SBATCH --constraint="apu"    # Remove or modify this line if a specific GPU type is not required

# Resources
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24    # Adjust according to how many CPUs your GPU job can use efficiently
#SBATCH --time=00:10:00        # Adjusted for longer GPU job

# Email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de


set -e
SECONDS=0


module purge
module load clang/18 gcc/14 rocm/6.3 openmpi/5.0 hdf5-mpi/1.14.1
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/projection_plots.py  --N_procs $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
