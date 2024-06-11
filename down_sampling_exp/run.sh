#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
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
# <<< conda initialize <<<

cd ~/cleanrl/down_sampling_exp

# Activate the conda environment
conda activate atari

# Ensure Poetry is available
export PATH="$HOME/.local/bin:$PATH"

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <resolution_width> <resolution_height> <total_timesteps>"
    exit 1
fi

# Set the variables from the command line arguments
RESOLUTION_WIDTH=$1
RESOLUTION_HEIGHT=$2
TOTAL_TIMESTEPS=$3

# Export the variables to make them available to the batch script
export RESOLUTION_WIDTH
export RESOLUTION_HEIGHT
export TOTAL_TIMESTEPS

# Submit the batch file
sbatch ~/cleanrl/down_sampling_exp/job1.batch
