#!/bin/bash
# us presidential nojoin
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy nojoin wordnet_big
# companies nojoin
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy nojoin wordnet_big
# movies nojoin
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy nojoin wordnet_big

# presidential left no agg
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left wordnet_big
# company left no agg
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left wordnet_big
# movies left no agg
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left wordnet_big

# presidential left dedup
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left --aggregation dedup wordnet_big
# companies left dedup
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left --aggregation dedup wordnet_big
# movies left dedup
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left --aggregation dedup wordnet_big

# presidential left dfs
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left --aggregation dfs wordnet_big
# companies left dfs
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left --aggregation dfs wordnet_big
# movies left dfs
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 --top_k 0  --join_strategy left --aggregation dfs wordnet_big