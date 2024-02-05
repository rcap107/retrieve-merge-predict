# %%
# cd ..

# # %%
# %load_ext autoreload
# %autoreload 2

# %%
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

import src.pipeline as utils
from src._join_aggregator import JoinAggregator
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import MetadataIndex, RawDataset


# %%
def metric_numerical(col, y):
    is_null = col.is_null()
    if sum(is_null) == len(col):
        return 0
    filled_null = col.fill_null(strategy="mean")
    _, p_nulls = fs.f_regression(is_null.to_numpy().reshape(-1, 1), y)
    _, p_filled = fs.f_regression(filled_null.to_numpy().reshape(-1, 1), y)
    return np.mean([-np.log(p_nulls), -np.log(p_filled)])


def metric_correlation(col, y):
    is_null = col.is_null()
    if sum(is_null) == len(col):
        return 0
    filled_null = col.fill_null(strategy="mean")
    tmp = pl.DataFrame([filled_null, pl.Series(y)])
    corr = tmp.select(pl.corr(*tmp.columns)).item()
    return corr


def metric_mutual_info_regression(col, y):
    col = (
        col.fill_null("null")
        .cast(pl.Categorical)
        .cast(pl.Int16)
        .to_numpy()
        .reshape(-1, 1)
    )
    m = fs.mutual_info_regression(col, y)
    return m[0]


def metric_discrete(col, y):
    is_null = col.is_null().to_numpy()
    if sum(is_null) == len(col):
        return 0
    filled_null = col.fill_null("null").cast(pl.Categorical).cast(pl.Int16).to_numpy()

    # col = col.fill_null("null").cast(pl.Categorical).cast(pl.Int16).to_numpy().reshape(-1, 1)
    _, p_nulls = fs.f_classif(is_null.reshape(-1, 1), y)
    _, p_filled = fs.f_classif(filled_null.reshape(-1, 1), y)
    if p_filled == 0:
        p_filled = [1]
    if p_nulls == 0:
        p_nulls = [1]

    return np.mean([-np.log(p_nulls), -np.log(p_filled)])


def metric_selection(merged_df: pl.DataFrame, target_columns: pl.Series, y: pl.Series):
    metrics_dict = {
        "numeric": [],
        "discrete": [],
    }
    num_cols = merged_df.select(cs.numeric() & cs.by_name(target_columns))
    cat_cols = merged_df.select(cs.string() & cs.by_name(target_columns))

    for col in num_cols:
        metrics_dict["numeric"].append((col.name, metric_numerical(col, y)))

    for col in cat_cols:
        metrics_dict["discrete"].append((col.name, metric_discrete(col, y)))
    return metrics_dict


# %% [markdown]
# # Extracting candidates from precomputed indices

# %%
yadl_version = "wordnet_big_num_cp"
metadata_dir = Path(f"data/metadata/{yadl_version}")
metadata_index_path = Path(f"data/metadata/_mdi/md_index_{yadl_version}.pickle")
index_dir = Path(f"data/metadata/_indices/{yadl_version}")

query_tab_path = Path("data/source_tables/us-accidents-yadl-ax.parquet")

base_table = pl.read_parquet(query_tab_path)
tab_name = query_tab_path.stem
mdata_index = MetadataIndex(index_path=metadata_index_path)

top_k = 10
selected_indices = ["minhash"]
indices = utils.load_indices(
    index_dir, selected_indices=selected_indices, tab_name=tab_name
)
minhash_index = indices["minhash"]
query_tab_metadata = RawDataset(
    query_tab_path.resolve(), "queries", "data/metadata/queries"
)
query_tab_metadata.save_metadata_to_json()
query_column = "col_to_embed"
query = base_table[query_column].drop_nulls()
query_results = minhash_index.query_index(query)
candidates = utils.generate_candidates(
    "minhash",
    query_results,
    mdata_index,
    query_tab_metadata.metadata,
    query_column,
    top_k,
)

# %%
r_list = []
for hash_, candidate_join in candidates.items():
    src_md, cnd_md, left_on, right_on = candidate_join.get_join_information()
    src_df = pl.read_parquet(src_md["full_path"])
    cnd_df = pl.read_parquet(cnd_md["full_path"])
    cols_to_agg = [col for col in cnd_df.columns if col not in right_on]

    ja = JoinAggregator(tables=[(cnd_df, right_on, cols_to_agg)], main_key=left_on)
    merged = ja.fit_transform(src_df)
    y = src_df["target"].to_numpy()
    target_columns = [
        col
        for col in merged.columns
        if col not in src_df.columns and col not in (left_on + right_on)
    ]
    results = metric_selection(merged, target_columns, y)
    for r_type, r_values in results.items():
        for e in r_values:
            entry = (cnd_md["df_name"], r_type, *e)
            r_list.append(dict(zip(["df_name", "type", "col_name", "metric"], entry)))
    break

# %%
merged

# %%
results = pl.from_dicts(r_list)

# %%
results

# %%
selected_columns = (
    results.filter(pl.col("type") == "numeric")
    .top_k(10, by="metric")
    .select(pl.col("col_name"))
)

# %%
selected_columns

# %%
"hasLongitude_mean" in selected_columns["col_name"]

# %%
[col for col in target_columns if col not in selected_columns["col_name"]]

# %%
with pl.Config(fmt_str_lengths=100):
    display(
        results.filter((pl.col("type") == "discrete") & (pl.col("metric") > 0)).sort(
            "metric", descending=True
        )
    )

# %%


# %%
