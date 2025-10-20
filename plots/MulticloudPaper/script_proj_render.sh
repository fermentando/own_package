#!/bin/bash -l
#SBATCH -o ./render.%j.log
#SBATCH -e ./render.%j.log
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J render_test
#SBATCH --nodes=1         
#SBATCH --ntasks-per-node=40    

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de
#SBATCH --time=00:10:00


set -e
SECONDS=0

module purge
module load ffmpeg/4.4 cuda/12.6 
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/render_example.py --n_jobs $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
