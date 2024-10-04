#!/bin/bash
#
#SBATCH --job-name=retrieve_merge_predict_realmlp
#SBATCH --output=res_realmlp_%A%a.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=parietal,normal,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --error error_%A_a.out


srun python main.py --input_path config/evaluation/general/config-vldb.toml
