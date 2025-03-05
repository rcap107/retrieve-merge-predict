#!/bin/bash

# Define the lists
data_lakes=("open_data_us" "wordnet_base" "wordnet_vldb_10" "wordnet_vldb_50" "binary_update")
retrieval_methods=("exact_matching" "minhash")

# Iterate over each combination of DATA_LAKE and RETRIEVAL_METHOD
for data_lake in "${data_lakes[@]}"; do
  for retrieval_method in "${retrieval_methods[@]}"; do
    echo "Running with DATA_LAKE=${data_lake} and RETRIEVAL_METHOD=${retrieval_method}"
    
    # Run the Python script with the current combination
    python profile_retrieval.py --data_lake_version "$data_lake" --retrieval_method "$retrieval_method" --n_iter 10
  done
done