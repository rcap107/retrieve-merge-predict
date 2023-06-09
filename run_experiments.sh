#!/bin/bash

##### WORDNET

# company left first
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation first wordnet_big
# companies left mean
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation mean wordnet_big
# companies left dfs
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dfs wordnet_big

# # us accidents left first
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation first wordnet_big
# us accidents left mean
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation mean wordnet_big
# us accidents left dfs
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation dfs wordnet_big

# presidential left first
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation first wordnet_big
# presidential left mean
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation mean wordnet_big
# presidential left dfs
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dfs wordnet_big

##### BINARY

# company left first
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation first binary
# companies left mean
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation mean binary
# companies left dfs
python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dfs binary

# # us accidents left first
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation first binary
# us accidents left mean
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation mean binary
# us accidents left dfs
python main_pipeline.py --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation dfs binary

# presidential left first
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation first binary
# presidential left mean
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation mean binary
# presidential left dfs
python main_pipeline.py --source_table_path data/source_tables/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dfs binary
