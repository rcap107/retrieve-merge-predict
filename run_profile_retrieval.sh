#!/bin/bash
#
#SBATCH --job-name=test_job
#SBATCH --output=res_test_job_%A_%a.txt
#
#SBATCH --ntasks=5
#SBATCH --time=10:00:00          
#SBATCH --cpus-per-task=32
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out
#
#SBATCH --array=0-5

# Define the lists
data_lakes=("open_data_us" "wordnet_base" "wordnet_vldb_10" "wordnet_vldb_50" "binary_update")
retrieval_methods=("exact_matching" "minhash")

srun python profile_retrieval.py --data_lake_version ${data_lake[$SLURM_ARRAY_TASK_ID]} --retrieval_method exact_matching --n_iter 3
sleep 5

