#!/bin/bash

# Check if the first conda path exists
if [ -d "/home-mscluster/sjumoorty2/anaconda3" ]; then
    # Setup for the first conda installation
    __conda_setup="$('/home-mscluster/sjumoorty2/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home-mscluster/sjumoorty2/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/home-mscluster/sjumoorty2/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/home-mscluster/sjumoorty2/anaconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
# If the first path doesn't exist, check the second conda path
elif [ -d "/home-mscluster/mdawood/miniconda3" ]; then
    # Setup for the second conda installation
    __conda_setup="$('/home-mscluster/mdawood/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home-mscluster/mdawood/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/home-mscluster/mdawood/miniconda3/etc/profile.d/conda.sh"
        else
            export PATH="/home-mscluster/mdawood/miniconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
else
    echo "Error: Neither conda installation found. Please check your conda paths."
    exit 1
fi

cd ~/cleanrl/grey_scale_exp

# Activate the conda environment
conda activate atari

# Ensure Poetry is available
export PATH="$HOME/.local/bin:$PATH"

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <use_grayscale> <total_timesteps>"
    exit 1
fi

# Set the variables from the command line arguments
USE_GRAYSCALE=$1
TOTAL_TIMESTEPS=$2

# Export the variables to make them available to the batch script
export USE_GRAYSCALE
export TOTAL_TIMESTEPS

# Submit the batch file
sbatch ~/cleanrl/grey_scale_exp/job1.batch
