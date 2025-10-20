#!/bin/bash -l
#SBATCH -o ./sep.%j.log
#SBATCH -e ./sep.%j.log
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J cloud_sep
#SBATCH --nodes=1         
#SBATCH --ntasks-per-node=40    

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de
#SBATCH --time=00:05:00

set -e
SECONDS=0

module purge
module load ffmpeg/4.4 cuda/12.6 
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/cloud_separation.py #$SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
