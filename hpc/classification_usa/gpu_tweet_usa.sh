#!/bin/bash

# SBATCH script to be executed
SBATCH_SCRIPT="process_tweets_gpu_gguf_usa.sbatch"

# Source directory for .csv.gz files
TWEET_DIR="/n/netscratch/cga/Lab/xiaokang/tweets_us_census"

# General log directory within geotweets
LOG_DIR="/n/netscratch/cga/Lab/anasuto/immigration/logs_usa"
FILES_COMPLETED_LOG="${LOG_DIR}/files_complete_log.txt"

# Directories for stdout and stderr
STDOUT_DIR="${LOG_DIR}/tweets_stdout"
STDERR_DIR="${LOG_DIR}/tweets_stderr"

# Ensure the log directories exist
mkdir -p "${LOG_DIR}" "${STDOUT_DIR}" "${STDERR_DIR}"

# Ensure the completed log file exists
touch "${FILES_COMPLETED_LOG}"

# Year, month, number of jobs, and time limit to process (required)
YEAR_TO_PROCESS=$1
MONTH_TO_PROCESS=$2
NUM_JOBS=$3
TIME_LIMIT=$4

if [[ -z "$YEAR_TO_PROCESS" || -z "$NUM_JOBS" || -z "$TIME_LIMIT" ]]; then
    echo "Usage: bash gpu_tweet_jobs.sh <year> [month] <num_jobs> <time_limit>"
    exit 1
fi

echo "Starting job submission for year: $YEAR_TO_PROCESS, month: ${MONTH_TO_PROCESS:-all}, num_jobs: $NUM_JOBS, time_limit: $TIME_LIMIT"

# Determine the files to process
declare -a FILES
if [[ -z "$MONTH_TO_PROCESS" ]]; then
    mapfile -t FILES < <(find "${TWEET_DIR}/${YEAR_TO_PROCESS}" -type f -name "${YEAR_TO_PROCESS}_*-tl_2021_*_tabblock20.parquet" | sort)
else
    mapfile -t FILES < <(find "${TWEET_DIR}/${YEAR_TO_PROCESS}" -type f -name "${YEAR_TO_PROCESS}_${MONTH_TO_PROCESS}_*-tl_2021_*_tabblock20.parquet" | sort)
fi

# Ensure files are found
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files found for the specified criteria. Exiting."
    exit 1
fi

# Counter for jobs submitted
JOB_COUNT=0



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

    # Submit the SLURM job for each parquet file and capture the job ID
    JOB_ID=$(sbatch --requeue --time=${TIME_LIMIT} --output=${STDOUT_DIR}/${BASENAME}.stdout.txt --error=${STDERR_DIR}/${BASENAME}.stderr.txt ${SBATCH_SCRIPT} ${FILE} | awk '{print $4}')
    
    echo "Current job id: ${JOB_ID}"
    
    # Increment job count and check if max jobs reached
    ((JOB_COUNT++))
    if [[ $JOB_COUNT -ge $NUM_JOBS ]]; then
        echo "Reached the maximum number of jobs (${NUM_JOBS}). Exiting."
        break
    fi
    
    # Sleep briefly to be kind to the scheduler
    sleep 1

done