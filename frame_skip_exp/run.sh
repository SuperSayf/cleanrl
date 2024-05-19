#!/bin/bash

# This script is used to submit the batch files to the cluster

frame_skip_exp_0=0
frame_skip_exp_1=2
frame_skip_exp_2=4
frame_skip_exp_3=6
frame_skip_exp_4=8

job_name_0=Atari_0
job_name_1=Atari_1
job_name_2=Atari_2
job_name_3=Atari_3
job_name_4=Atari_4

total_timestepstotal_timesteps=10000000

bash slurm.sh $frame_skip_exp_0 $job_name_0 $total_timesteps
# sbatch slurm.batch $frame_skip_exp_1 $job_name_1 $total_timesteps
# sbatch slurm.batch $frame_skip_exp_2 $job_name_2 $total_timesteps
# sbatch slurm.batch $frame_skip_exp_3 $job_name_3 $total_timesteps
# sbatch slurm.batch $frame_skip_exp_4 $job_name_4 $total_timesteps