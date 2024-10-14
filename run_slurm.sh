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
#
#SBATCH --array=0-3


CONFIG_RUNS=(config-vldb-yadlbase.toml config-vldb-yadl10k.toml config-vldb-yadl50k.toml config-vldb-binary.toml)

srun python main.py --input_path config/evaluation/general/${CONFIG_RUNS[$SLURM_ARRAY_TASK_ID]}

sleep 5
