#!/bin/bash -l
#SBATCH -o ./stdout_files/analysis.%j.log
#SBATCH -e ./stdout_files/analysis.%j.log
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J quick_analysis
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de
#SBATCH --time=00:20:00
   

set -e
SECONDS=0

module purge
module load ffmpeg/4.4 
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/cloud_separation.py  --N_procs $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Rough analysis completed."
