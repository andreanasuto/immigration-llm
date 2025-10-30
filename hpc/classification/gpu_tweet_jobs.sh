#!/bin/bash

# SBATCH script to be executed
SBATCH_SCRIPT="process_tweets_gpu_gguf.sbatch"

# Source directory for .csv.gz files
TWEET_DIR="/n/holylabs/LABS/cga/Lab/data/geo-tweets/cga-sbg-tweets"

# General log directory within geotweets
LOG_DIR="/n/netscratch/cga/Lab/anasuto/immigration/logs" # or "/n/home03/anasuto/geotweets/logs"
FILES_COMPLETED_LOG="${LOG_DIR}/files_complete_log.txt"

# Directories for stdout and stderr
STDOUT_DIR="${LOG_DIR}/tweets_stdout"
STDERR_DIR="${LOG_DIR}/tweets_stderr"

# Year and month to process (required)
YEAR_TO_PROCESS=$1
MONTH_TO_PROCESS=$2

if [[ -z "$YEAR_TO_PROCESS" ]]; then
    echo "Usage: bash gpu_tweet_jobs.sh <year> [month]"
    exit 1
fi

echo "Starting job submission for year: $YEAR_TO_PROCESS, month: ${MONTH_TO_PROCESS:-all}"

# Determine the files to process
declare -a FILES
if [[ -z "$MONTH_TO_PROCESS" ]]; then
    # Process all months in the given year, starting from the earliest month available
    mapfile -t FILES < <(find "${TWEET_DIR}/${YEAR_TO_PROCESS}" -type f -name "${YEAR_TO_PROCESS}_*.csv.gz" | sort)
else
    # Process only the specified month
    mapfile -t FILES < <(find "${TWEET_DIR}/${YEAR_TO_PROCESS}" -type f -name "${YEAR_TO_PROCESS}_${MONTH_TO_PROCESS}_*.csv.gz" | sort)
fi

# Ensure files are found
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files found for the specified criteria. Exiting."
    exit 1
fi

# Loop through the determined files and submit jobs
for FILE in "${FILES[@]}"; do
    BASENAME=$(basename ${FILE})
    
    # Check if the file is already in the completed log
    if grep -Fxq "$BASENAME" "$FILES_COMPLETED_LOG"; then
        echo "${BASENAME} has already been processed. Skipping."
        continue
    fi
    
    # Print filename as a progress indicator
    echo "Submitting job for ${BASENAME}"

    # Submit the SLURM job for each .csv.gz file and capture the job ID
    JOB_ID=$(sbatch --requeue --output=${STDOUT_DIR}/${BASENAME}.stdout.txt --error=${STDERR_DIR}/${BASENAME}.stderr.txt ${SBATCH_SCRIPT} ${FILE} | awk '{print $4}')
    
    echo "Current job id: ${JOB_ID}"

    # Sleep briefly to be kind to the scheduler
    sleep 1

done