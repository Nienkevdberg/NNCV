#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00


cd $HOME/NNCV/Final assignment
echo "Running from directory: $(pwd)"

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh