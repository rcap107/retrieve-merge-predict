#!/bin/bash
#
#SBATCH --job-name=linear
#SBATCH --output=res_linear_%A%a.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=parietal,normal
#SBATCH --cpus-per-task=32
#SBATCH --error error_%A_a.out

srun python main.py --input_path config/evaluation/general/config-vldb.toml

sleep 5
