#!/bin/bash -l
#SBATCH -o ./analysis.%j.log
#SBATCH -e ./analysis.%j.log
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J quick_analysis
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=240000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=fernando@mpa-garching.mpg.de
#SBATCH --time=00:10:00


set -e
SECONDS=0

module purge
module load ffmpeg/4.4 
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /u/ferhi/own_package/plots/projection_plots.py  --N_procs $SLURM_CPUS_PER_TASK
srun python /u/ferhi/own_package/plots/mass_evolution.py --N_procs $SLURM_CPUS_PER_TASK
srun python /u/ferhi/own_package/plots/entrainment.py --N_procs $SLURM_CPUS_PER_TASK
srun python /u/ferhi/own_package/plots/2dhist.py --N_procs $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Rough analysis completed."
