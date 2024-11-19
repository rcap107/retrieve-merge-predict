# %%
import json

import polars as pl
from sklearn.model_selection import ParameterGrid


#
def prepare_config(config_dict):
    pars = ParameterGrid(config_dict)
    df_config = pl.from_dicts(list(pars))
    return df_config


# %%
df = pl.read_parquet("results/master_list.parquet")
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
df_test.write_parquet("results/temp_results_general.parquet")
# %%
# Retrieval method configuration (Starmie, no 50k/open data)
config_general = json.load(
    open("config/required_configurations/config-retrieval_comparison.json")
)
df_config = prepare_config(config_general)
group_keys = df_config.columns
df_test = df_config.join(df, on=group_keys, how="inner")
df_test.write_parquet("results/temp_results_retrieval.parquet")

# %%
# Aggregation
config_general = json.load(
    open("config/required_configurations/config-aggregation_comparison.json")
)
df_config = prepare_config(config_general)
group_keys = df_config.columns
df_test = df_config.join(df, on=group_keys, how="inner")
df_test.write_parquet("results/temp_results_aggregation.parquet")

# %%
