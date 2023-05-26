#!/bin/bash
python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 wordnet

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 wordnet

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 wordnet

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 binary

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 binary

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 binary

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 seltab

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 seltab

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 seltab

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 full

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/company-employees/company-employees-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 full

python main_pipeline.py --source_table_path data/source_tables/ken_datasets/the-movies-dataset/movies-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 1000 full
