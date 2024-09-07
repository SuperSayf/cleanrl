#!/bin/bash

# Try the original conda setup
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

# If the above fails, try the alternative setup
if ! command -v conda &> /dev/null; then
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
fi

cd ~/cleanrl/combined_exp

# Activate the conda environment
conda activate atari

# Ensure Poetry is available
export PATH="$HOME/.local/bin:$PATH"

# Default values
USE_GRAYSCALE=false
FRAME_SKIP=0
USE_COMPRESSION=false
JPEG_QUALITY=95
RESOLUTION_WIDTH=84
RESOLUTION_HEIGHT=84
TOTAL_TIMESTEPS=10000000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --grayscale) USE_GRAYSCALE=true; shift ;;
        --frame-skip) FRAME_SKIP=$2; shift 2 ;;
        --compression) USE_COMPRESSION=true; JPEG_QUALITY=$2; shift 2 ;;
        --resolution) RESOLUTION_WIDTH=$2; RESOLUTION_HEIGHT=$3; shift 3 ;;
        --total-timesteps) TOTAL_TIMESTEPS=$2; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Export the variables to make them available to the batch script
export USE_GRAYSCALE
export FRAME_SKIP
export USE_COMPRESSION
export JPEG_QUALITY
export RESOLUTION_WIDTH
export RESOLUTION_HEIGHT
export TOTAL_TIMESTEPS

# Submit the batch file
sbatch ~/cleanrl/combined_exp/job1.batch