#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <frame_skip> <total_timesteps>"
    exit 1
fi

# Set the variables from the command line arguments
FRAME_SKIP=$1
TOTAL_TIMESTEPS=$2

# Export the variables to make them available to the batch script
export FRAME_SKIP
export TOTAL_TIMESTEPS

# Submit the batch file
sbatch job1.batch
