#!/bin/bash
python main_pipeline.py --metadata_dir data/metadata/binary --metadata_index data/metadata/mdi/md_index_binary.pickle --index_dir data/metadata/indices \
--source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 10

python main_pipeline.py --metadata_dir data/metadata/binary --metadata_index data/metadata/mdi/md_index_binary.pickle --index_dir data/metadata/indices \
--source_table_path data/source_tables/ken_datasets/presidential-results/presidential-results-prepared.parquet \
--query_column col_to_embed --sampling_seed 42 --iterations 10