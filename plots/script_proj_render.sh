#!/bin/sh
#
#SBATCH -J parpl
#SBATCH -o parpl."%j".out
#SBATCH -e parpl."%j".out
#SBATCH --mail-user fernando@mpa-garching.mpg.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:25:00

set -e
SECONDS=0

module purge
module load ffmpeg/4.4
module list

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF5_HOME/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FFTW_HOME/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$I_MPI_ROOT/intel64/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$I_MPI_ROOT/intel64/lib/release/

# set number of OMP threads *per process*
export OMP_NUM_THREADS=1

srun python /u/ferhi/Scripts/plots/render_example.py $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
