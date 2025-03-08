#!/bin/sh
#
#SBATCH -J generation_ics
#SBATCH -o generation_ics."%j".out
#SBATCH -e generation_ics."%j".out
#SBATCH --mail-user fernando@mpa-garching.mpg.de
#SBATCH --mail-type=ALL

#SBATCH -p p.test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:10:00

set -e
SECONDS=0

module purge

# set number of OMP threads *per process*
export OMP_NUM_THREADS=1

# Run the Python script with the number of CPUs
srun python /u/ferhi/Scripts/athenapk/generate_ics.py --n_jobs $SLURM_CPUS_PER_TASK


echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
