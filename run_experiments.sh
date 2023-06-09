#!/bin/bash


# company left 
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation first wordnet_big
# companies left dedup
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dedup wordnet_big
# companies left dfs
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dfs wordnet_big

# # us accidents left noagg
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/us-accidents/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation first wordnet_big
# us accidents left dedup
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/us-accidents/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation dedup wordnet_big
# us accidents left dfs
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/us-accidents/us-accidents-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500 --top_k 50  --join_strategy left --aggregation dfs wordnet_big

# presidential left no agg
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation first wordnet_big
# presidential left dedup
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dedup wordnet_big
# presidential left dfs
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/us-presidential-results-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 500  --top_k 50  --join_strategy left --aggregation dfs wordnet_big
