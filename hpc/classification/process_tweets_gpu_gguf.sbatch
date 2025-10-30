#!/bin/bash
#SBATCH --job-name=process_tweets         # Job name
#SBATCH --output=/n/netscratch/cga/Lab/anasuto/immigration//logs/tweets_stdout/tweet_gguf_%j.out    # Standard output log - overwritten when .sh script runs
#SBATCH --error=/n/netscratch/cga/Lab/anasuto/immigration/logs/tweets_stderr/tweet_gguf_%j.err     # Error log file - overwritten when .sh script runs
#SBATCH --ntasks=1                        # Number of tasks (1 task for this script)
#SBATCH --cpus-per-task=1                 # Request 1 CPU core
#SBATCH --mem=50G                         # Request 25GB of memory
#SBATCH --time=00:30:00                   # Set a time limit of 11 minutes
#SBATCH --partition=gpu_requeue           # Partition name for GPU jobs
#SBATCH --gres=gpu:1                      # Request 1 GPU SBATCH --constraint="h100|a100" Constrain to nodes with H100 or A100 GPUs

# Load necessary Python module
#module load nvhpc/23.7-fasrc01
#module load cuda/12.2.0-fasrc01

# Trap to requeue the job if interrupted
#trap 'scontrol requeue $SLURM_JOB_ID' TERM INT HUP

# Directly specify the Python executable from the gpu environment
/n/home03/anasuto/.conda/envs/gpu/bin/python llama.cpp/classify_tweets_gguf.py ${1} ${2}