import argparse
import json
import logging
import os
import pickle
from difflib import SequenceMatcher
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import sklearn.feature_selection as fs
import sklearn.metrics as metrics
from polars import selectors as cs

import src.pipeline as pipeline
from src._join_aggregator import JoinAggregator
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import MetadataIndex, RawDataset
from src.utils.indexing import write_candidates_on_file

# Select the yadl version
yadl_version = "wordnet_big_num_cp"

# Prepare the metadata
metadata_dir = Path(f"data/metadata/{yadl_version}")
metadata_index_path = Path(f"data/metadata/_mdi/md_index_{yadl_version}.pickle")
mdata_index = MetadataIndex(index_path=metadata_index_path)

# Prepare the index dir
index_dir = Path(f"data/metadata/_indices/{yadl_version}")

# Prepare the query table
query_tab_path = Path("data/source_tables/us-accidents-yadl-ax.parquet")

base_table = pl.read_parquet(query_tab_path)
tab_name = query_tab_path.stem

top_k = 10
selected_index = "minhash"
index_path = Path(index_dir, selected_index + "_index.pickle")
minhash_index = pipeline.load_index(index_path, tab_name)

query_tab_metadata = RawDataset(
    query_tab_path.resolve(), "queries", "data/metadata/queries"
)
query_tab_metadata.save_metadata_to_json()
query_column = "col_to_embed"
query = base_table[query_column].drop_nulls()
query_results = minhash_index.query_index(query)
candidates = pipeline.generate_candidates(
    "minhash",
    query_results,
    mdata_index,
    query_tab_metadata.metadata,
    query_column,
    top_k,
)
output_file_name = "joinpaths.txt"
write_candidates_on_file(candidates, output_file_name)

r_list = []
for hash_, candidate_join in candidates.items():
    src_md, cnd_md, left_on, right_on = candidate_join.get_join_information()
    src_df = pl.read_parquet(src_md["full_path"])
    cnd_df = pl.read_parquet(cnd_md["full_path"])
    cols_to_agg = [col for col in cnd_df.columns if col not in right_on]

    ja = JoinAggregator(tables=[(cnd_df, right_on, cols_to_agg)], main_key=left_on)
    y = src_df["target"].to_numpy()
    merged = ja.fit_transform(src_df, y=y)
    break
