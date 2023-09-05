#!/bin/bash

# presidential left first
python main_pipeline.py open_data --source_table_path data/source_tables/us-accidents-yadl.parquet \
--query_column County --sampling_seed 42 --iterations 50  --top_k 20  --join_strategy left --n_jobs 8 --aggregation first
