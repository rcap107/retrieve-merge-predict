#!/bin/bash

##### WORDNET

# company left first
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50 --n_jobs 8  --join_strategy left --aggregation first wordnet_big_num_cp
# companies left mean
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50 --n_jobs 8  --join_strategy left --aggregation mean wordnet_big_num_cp
# companies left dfs
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50 --n_jobs 8  --join_strategy left --aggregation dfs wordnet_big_num_cp

# # us accidents left first
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50 --n_jobs 8  --join_strategy left --aggregation first wordnet_big_num_cp
# us accidents left mean
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50 --n_jobs 8  --join_strategy left --aggregation mean wordnet_big_num_cp
# us accidents left dfs
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50 --n_jobs 8  --join_strategy left --aggregation dfs wordnet_big_num_cp

# presidential left first
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50 --n_jobs 8  --join_strategy left --aggregation first wordnet_big_num_cp
# presidential left mean
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50 --n_jobs 8  --join_strategy left --aggregation mean wordnet_big_num_cp
# presidential left dfs
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl-ax.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50 --n_jobs 8  --join_strategy left --aggregation dfs wordnet_big_num_cp

##### BINARY