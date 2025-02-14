#!/bin/bash
#SBATCH --job-name=validation             # Job name
#SBATCH --output=/n/netscratch/cga/Lab/anasuto/immigration/tests/logs/validation_test_%j.out    # Standard output log
#SBATCH --error=/n/netscratch/cga/Lab/anasuto/immigration/tests/logs/validation_test_%j.err     # Error log file
#SBATCH --ntasks=1                        # Number of tasks (1 task for this script)
#SBATCH --cpus-per-task=1                 # Request 2 CPU core
#SBATCH --mem=40G                         # Request 25GB of memory
#SBATCH --time=02:00:00                   # Set a time limit of 11 minutes
#SBATCH --partition=gpu_test              # Partition name for GPU jobs
#SBATCH --gres=gpu:1                      # Request 1 GPU SBATCH --constraint=a100|h100            # Restrict job to A100 or H100 GPUs


# Load necessary Python module
#module load python/3.10.13-fasrc01

# Directly specify the Python executable from the gpu environment
/n/home03/anasuto/.conda/envs/gpu/bin/python validation_all.py # add "$1" when using validation.py