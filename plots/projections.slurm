#!/bin/bash -l
#SBATCH -o ./proj.%j.log
#SBATCH -e ./proj.%j.log
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J quick_projections
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=40    # Ensure SLURM does not interfere with threading
#SBATCH --ntasks=1         # Run 40 parallel Python instances (one per task)

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de
#SBATCH --time=00:02:00


set -e
SECONDS=0

module purge
module load ffmpeg/4.4 cuda/12.6 
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/projection_plots.py  --N_procs $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"
