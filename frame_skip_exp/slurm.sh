#!/bin/bash

# Check if frame skip value, job name, and total timesteps are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Frame skip value, job name, and/or total timesteps not provided."
    echo "Usage: $0 <frame_skip_value> <job_name> <total_timesteps>"
    exit 1
fi

# Set the frame skip value, job name, and total timesteps
frame_skip="$1"
job_name="$2"
total_timesteps="$3"

#SBATCH --job-name=${job_name}
#SBATCH --output="/home-mscluster/sjumoorty2/atari/cleanrl/node_output/node_output_%j.txt"
#SBATCH --ntasks=5
#SBATCH --nodes=5
#SBATCH --time=3-00:00:00
#SBATCH --partition=bigbatch

for i in $(seq 0 $((SLURM_NTASKS-1))); do
    exp_name="frameskip-val-${frame_skip}-task-N${i}"
    srun --exclusive -N1 -n1 poetry run python cleanrl/dqn_atari_jax.py --exp-name ${exp_name} --env-id ALE/Breakout-v5 --frame_skip ${frame_skip} --total-timesteps ${total_timesteps} --track --wandb-project-name atari &
done
wait