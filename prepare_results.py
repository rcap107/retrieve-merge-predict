# %%
import json

import polars as pl
from sklearn.model_selection import ParameterGrid

from src.utils import constants


#
def prepare_config(config_dict):
    pars = ParameterGrid(config_dict)
    df_config = pl.from_dicts(list(pars))
    return df_config


def fix_duplicate_runs(df):
    _df = df.group_by(constants.GROUPING_KEYS).agg(pl.all().last())
    return _df


# %%
df = pl.read_parquet("results/master_list.parquet")
df = fix_duplicate_runs(df)
#%%
df = df.with_columns(base_table=pl.col("base_table").str.split("-").list.first())
# %%
# General configuration (all data lakes, no Starmie)
config_general = json.load(
    open("config/required_configurations/config-generic_comparison.json")
)
df_config = prepare_config(config_general)
group_keys = df_config.columns
df_test = df_config.join(df, on=group_keys, how="inner")
df_test.write_parquet("results/results_general.parquet")
# %%
# Retrieval method configuration (Starmie, no 50k/open data)
config_general = json.load(
    open("config/required_configurations/config-retrieval_comparison.json")
)
df_config = prepare_config(config_general)
group_keys = df_config.columns
df_test = df_config.join(df, on=group_keys, how="inner")
df_test.write_parquet("results/results_retrieval.parquet")

# %%
# Aggregation
config_general = json.load(
    open("config/required_configurations/config-aggregation_comparison.json")
)
df_config = prepare_config(config_general)
group_keys = df_config.columns
df_test = df_config.join(df, on=group_keys, how="inner")
df_test.write_parquet("results/results_aggregation.parquet")

# %%
