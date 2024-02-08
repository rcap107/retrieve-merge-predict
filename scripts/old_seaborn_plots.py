# %%


# %%
# %cd ~/bench

# %%
# %load_ext autoreload
# %autoreload 2
import json
import tarfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec

import src.utils.plotting as plotting
from src.utils.logging import read_logs

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

# %% [markdown]
# # Combine all results

# %%
root_path = Path("results/big_batch")
df_list = []
for rpath in root_path.iterdir():
    df_raw = read_logs(exp_name=None, exp_path=rpath)
    df_list.append(df_raw)

df_results = pl.concat(df_list)

# %% [markdown]
# # BIG BATCH

# %%
df_ = df_results.select(
    pl.col(
        [
            "scenario_id",
            "target_dl",
            "jd_method",
            "base_table",
            "estimator",
            "chosen_model",
            "aggregation",
            "r2score",
            "time_fit",
            "time_predict",
            "time_run",
            "epsilon",
        ]
    )
).filter(
    (~pl.col("base_table").str.contains("open_data"))
    & (pl.col("target_dl") != "wordnet_big")
)

# %%
df_ = df_.with_columns(
    (pl.col("jd_method") + " | " + pl.col("target_dl")).alias("case")
)

# %%
col_order = (
    df_.select(pl.col("base_table")).unique().sort("base_table").to_series().to_list()
)

# %%
df_ = df_.group_by(
    ["target_dl", "jd_method", "base_table", "estimator", "chosen_model"]
).map_groups(lambda x: x.with_row_count("fold_id"))

# %%
joined = df_.join(
    df_.filter(pl.col("estimator") == "nojoin"),
    on=["target_dl", "jd_method", "base_table", "chosen_model", "fold_id"],
    how="left",
).with_columns((pl.col("r2score") - pl.col("r2score_right")).alias("difference"))

# %%
results_full = joined.filter(~pl.col("base_table").str.contains("depleted"))
results_depleted = joined.filter(pl.col("base_table").str.contains("depleted"))

# %% [markdown]
# # Early Seaborn plots

# %%
with sns.axes_style("whitegrid"):
    ax = sns.catplot(
        data=df_.filter(~pl.col("base_table").str.contains("open_data")).to_pandas(),
        # data=df_.filter(pl.col("chosen_model") == "catboost").to_pandas(),
        x="r2score",
        y="estimator",
        hue="case",
        kind="box",
        col="base_table",
        row="chosen_model",
        sharex=False,
        # col_wrap=2,
        # col_order=col_order,
        # facet_kws={"xlim": [0,1]}
        palette="tab20",
    )

# %%
with sns.axes_style("whitegrid"):
    ax = sns.catplot(
        data=joined.filter(~pl.col("base_table").str.contains("open_data")).to_pandas(),
        # data=df_.filter(pl.col("chosen_model") == "catboost").to_pandas(),
        x="difference",
        y="estimator",
        hue="case",
        kind="box",
        col="base_table",
        row="chosen_model",
        sharex=True,
        # col_wrap=2,
        # col_order=col_order,
        # facet_kws={"xlim": [0,1]}
        palette="tab20",
    )

# %%
for table in df_["base_table"].unique().sort():
    ax = sns.catplot(
        data=df_.filter(pl.col("base_table") == table).to_pandas(),
        y="r2score",
        x="estimator",
        hue="case",
        kind="box",
        # col="base_table",
        col="chosen_model",
        # sharex=True,
        # col_wrap=2,
        # col_order=col_order,
        # facet_kws={"xlim": [0,1]}
        palette="tab20",
    )
    ax.set_xticklabels(rotation=30)
    ax.fig.subplots_adjust(top=0.9)

    ax.fig.suptitle(table)

# %%
for table in joined["base_table"].unique().sort():
    ax = sns.catplot(
        data=joined.filter(pl.col("base_table") == table).to_pandas(),
        y="difference",
        x="estimator",
        hue="case",
        kind="box",
        # col="base_table",
        col="chosen_model",
        # sharex=True,
        # col_wrap=2,
        # col_order=col_order,
        # facet_kws={"xlim": [0,1]}
        palette="tab20",
    )
    ax.set_xticklabels(rotation=30)
    ax.fig.subplots_adjust(top=0.9)

    ax.fig.suptitle(table)
