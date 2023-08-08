#!/bin/bash

python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 100  --top_k 20  --join_strategy left  --aggregation mean --selected_indices minhash  --n_jobs 1 wordnet_big_num_cp
