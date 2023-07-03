#!/bin/bash

python main_pipeline.py --source_table_path data/source_tables/company-employees-yadl.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 10  --top_k 5  --join_strategy left --n_jobs 1 --aggregation first binary
