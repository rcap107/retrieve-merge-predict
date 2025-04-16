# %%
import polars as pl
import pandas as pd

# %%
df_starmie = (
    pl.read_csv("stats_retrieval_starmie.csv")
    .drop("time_save")
    .with_columns(index_name=pl.lit("starmie"))
)
df_others = pl.read_csv("stats_retrieval_others.csv").drop("time_save", "n_candidates")
# %%
val_starmie = (
    df_starmie.group_by("data_lake_version", "base_table", "index_name")
    .agg(pl.mean("time_create", "time_load", "time_query"))
    .with_columns(total_query=pl.col("time_load") + pl.col("time_query"))["total_query"]
    .mean()
) / 6

# %% Include only the data lakes that starmie works on
_d = (
    df_others.filter(~pl.col("data_lake_version").is_in(["wordnet_vldb_50", "open_data_us"])).with_columns(
        total_query=pl.when(pl.col("index_name") == "exact_matching")
        .then(pl.sum_horizontal("time_create", "time_load", "time_query"))
        .otherwise(pl.sum_horizontal("time_load", "time_query"))
    )
    .group_by("index_name")
    .agg(pl.mean("total_query"))
    .with_columns(total_query=pl.col("total_query") / 6)
)

r_dict = dict(_d.rows())
r_dict["starmie"] = val_starmie

pl.from_dict({"jd_method": r_dict.keys(), "time_query": r_dict.values()}).write_csv(
    "avg_query_time_for_pareto_plot_retrieval.csv"
)
# %% Now all data lakes and no starmie
_d = (
    df_others.with_columns(
        total_query=pl.when(pl.col("index_name") == "exact_matching")
        .then(pl.sum_horizontal("time_create", "time_load", "time_query"))
        .otherwise(pl.sum_horizontal("time_load", "time_query"))
    )
    .group_by("index_name")
    .agg(pl.mean("total_query"))
    .with_columns(total_query=pl.col("total_query") / 6)
)

r_dict = dict(_d.rows())

pl.from_dict({"jd_method": r_dict.keys(), "time_query": r_dict.values()}).write_csv(
    "avg_query_time_for_pareto_plot_all_datalakes.csv"
)

# %%
df_others_max_ram=df_others.filter(~pl.col("data_lake_version").is_in(["wordnet_vldb_50", "open_data_us"])).with_columns(
        max_ram=pl.max_horizontal("peak_create", "peak_query")
    )
# %%
import seaborn as sns

sns.displot(data=df_others_max_ram.to_pandas(), x="max_ram", col="index_name", binwidth=200)