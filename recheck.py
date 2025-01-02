# %%

import json
import pickle
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import polars as pl
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.utils.logging import read_and_process, read_logs

pl.Config.set_fmt_str_lengths(150)
pl.Config.set_tbl_rows(30)


all_configurations = {
    "source_table": [
        "company_employees",
        "housing_prices",
        "us_elections",
        "us_accidents_2021",
        "us_accidents_large",
        "schools",
        "us_county_population",
    ],
    "target_dl": [
        "binary_update",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
        "wordnet_full",
        "open_data_us",
    ],
    "jd_method": ["minhash", "minhash_hybrid", "exact_matching", "starmie"],
    "chosen_model": ["catboost", "ridgecv", "realmlp", "resnet"],
    "aggregation": ["first", "mean", "dfs"],
    "estimator": [
        "full_join",
        "nojoin",
        "highest_containment",
        "best_single_join",
        "stepwise_greedy_join",
    ],
}


def prepare_config(config_dict):
    pars = ParameterGrid(config_dict)
    df_config = pl.from_dicts(list(pars))
    return df_config


df_all_config = prepare_config(all_configurations)

keys = df_all_config.columns

# preparing impossible configurations
impossible_starmie = df_all_config.filter(
    (pl.col("jd_method") == "starmie")
    & (pl.col("target_dl").is_in(["open_data_us", "wordnet_vldb_50k"]))
)

impossible_internal_tables = df_all_config.filter(
    ((pl.col("source_table") == "schools") & (~(pl.col("target_dl") == "open_data_us")))
    | (
        (pl.col("source_table") == "us_county_population")
        & ((pl.col("target_dl") == "open_data_us"))
    )
)

df_cleaned = df_all_config.join(impossible_starmie, on=keys, how="anti").join(
    impossible_internal_tables, on=keys, how="anti"
)

# %%
# Configurations I need
runs_starmie = df_cleaned.filter(
    (pl.col("aggregation") == "first")
    & ~(pl.col("target_dl").is_in(["open_data_us", "wordnet_vldb_50k"]))
)

runs_general = df_cleaned.filter(
    (pl.col("aggregation") == "first") & (pl.col("jd_method") != "starmie")
)

runs_aggr_1 = df_cleaned.filter(
    ((pl.col("jd_method") != "starmie"))
    & (
        (
            pl.col("estimator").is_in(
                ["nojoin", "best_single_join", "highest_containment"]
            )
        )
    )
)
runs_aggr_2 = df_cleaned.filter(
    (~pl.col("target_dl").is_in(["open_data_us", "wordnet_vldb_50k"]))
    & (
        (
            pl.col("estimator").is_in(
                ["nojoin", "best_single_join", "highest_containment"]
            )
        )
    )
)

# %%
df_overall = pl.read_csv("results/master_list.csv")


# %%
(
    runs_general.join(df_overall, on=keys, how="left")
    .filter(~pl.col("status").is_null())
    .group_by(keys)
    .agg(pl.len())
    .filter(pl.col("len") < 10)
    .sort("target_dl")
).write_csv("incomplete.csv")

# %%
(
    runs_general.join(df_overall, on=keys, how="left")
    .filter(pl.col("status").is_null())
    .group_by(keys)
    .agg(pl.len())
    .sort("target_dl")
)
# .write_csv("incomplete.csv")

# %%
