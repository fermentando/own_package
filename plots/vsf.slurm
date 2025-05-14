#!/bin/bash -l
#SBATCH -o ./vsf.%j
#SBATCH -e ./vsf.%j
#SBATCH --job-name=vsf
#SBATCH --constraint="apu"

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
##SBATCH --mem=240000

# Email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de
#SBATCH --time=00:30:00

set -e
SECONDS=0


module purge
module load clang/18 gcc/14 rocm/6.3 openmpi/5.0 hdf5-mpi/1.14.1
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/3D_vsf_hdf.py --N_procs $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
