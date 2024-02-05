# %%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2
# %%
import json
import tarfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.gridspec import GridSpec

import src.utils.constants as constants
import src.utils.plotting as plotting
from src.utils.logging import read_logs

# %%

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)
LABEL_MAPPING = constants.LABEL_MAPPING

sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")


# %%


def prepare_relative_times(df, grouping_variable):
    f = (
        df.group_by(grouping_variable)
        .agg(
            pl.col("time_prepare").mean(),
            pl.col("time_model_train").mean(),
            pl.col("time_join_train").mean(),
            pl.col("time_model_predict").mean(),
            pl.col("time_join_predict").mean(),
        )
        .with_columns(pl.sum_horizontal(cs.numeric()).alias("total"))
        .with_columns(cs.starts_with("time_") / pl.col("total"))
        .sort("total")
        .drop("total")
        .melt(id_vars=grouping_variable)
    )
    return f


def prepare_absolute_times(df, grouping_variable):
    f = (
        df.group_by(grouping_variable)
        .agg(
            pl.col("time_prepare").mean(),
            pl.col("time_model_train").mean(),
            pl.col("time_join_train").mean(),
            pl.col("time_model_predict").mean(),
            pl.col("time_join_predict").mean(),
        )
        .with_columns(pl.sum_horizontal(cs.numeric()).alias("total"))
        .sort("total")
        .drop("total")
        .melt(id_vars=grouping_variable)
    )
    return f


def prepare_total_times(df, grouping_variable):
    f = (
        df.group_by(grouping_variable)
        .agg(
            pl.col("time_prepare").mean(),
            pl.col("time_model_train").mean(),
            pl.col("time_join_train").mean(),
            pl.col("time_model_predict").mean(),
            pl.col("time_join_predict").mean(),
        )
        .with_columns(pl.sum_horizontal(cs.numeric()).alias("total"))
        .drop(cs.starts_with("time_"))
        .melt(id_vars=grouping_variable)
    )
    return f


def prepare_subplot(df, ax, grouping_variable, variant: str = "absolute"):
    if variant == "absolute":
        results = prepare_absolute_times(df, grouping_variable)
        label = "Execution time (s)"
    elif variant == "relative":
        results = prepare_relative_times(df, grouping_variable)
        label = "Frac. spent in sections"
    elif variant == "total":
        results = prepare_total_times(df, grouping_variable)
        label = "Execution time (s)"
    else:
        raise NotImplementedError

    to_concat = []
    for _, gr in results.sort("variable", descending=True).group_by(
        [grouping_variable], maintain_order=True
    ):
        new_g = (
            gr.sort("variable")
            .with_columns(pl.col("value").cum_sum().alias("csum"))
            .with_columns(
                pl.col("csum").alias("bottom").shift(1).fill_null(0),
            )
        )
        to_concat.append(new_g)
    df_c = pl.concat(to_concat)
    df_c = df_c.with_columns(
        pl.col(grouping_variable)
        .map_elements(constants.ORDER_MAPPING[grouping_variable].index)
        .alias("order")
    ).sort("order", descending=True)

    dicts = df_c.group_by(["variable"], maintain_order=True).agg(pl.all()).to_dicts()

    y_ticks = []
    y_tick_labels = []

    for idx, d in enumerate(dicts):
        print(d["variable"])
        p = ax.barh(
            y=d[grouping_variable],
            width=d["value"],
            left=d["bottom"],
            label=d["variable"],
            tick_label=d[grouping_variable],
        )
        if variant == "total":
            ax.bar_label(p, fmt="{:.2f}", label_type="edge", fontsize=12)
    ax.set_xlabel(label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_ticks = ax.get_yticks()
    print(y_ticks)
    ax.set_yticks(
        y_ticks, [LABEL_MAPPING[grouping_variable][v] for v in d[grouping_variable]]
    )


# %%
mapping_times = {
    "time_join_predict": "Predict(join)",
    "time_model_predict": "Predict(model)",
    "time_join_train": "Train(join)",
    "time_model_train": "Train(model)",
    "time_prepare": "Prepare",
}


df_raw = pl.read_parquet("results/overall/wordnet_general_first.parquet")
df_raw = df_raw.fill_null(0)
grouping_variable = "estimator"
fig, axs = plt.subplots(
    2,
    1,
    # layout="constrained",
    sharey=True,
    figsize=(4, 8),
)
# fig, axs = plt.subplots(1,2, layout="constrained", sharey=True, figsize=(8,2))
prepare_subplot(df_raw, axs[0], grouping_variable, "relative")
prepare_subplot(df_raw, axs[1], grouping_variable, "total")
# axs[2].axis("off")
h, l = axs[0].get_legend_handles_labels()
labels = [mapping_times[_] for _ in l]
fig.legend(
    h,
    labels,
    bbox_to_anchor=(0, 0.02, 1, 0.3),
    loc="lower left",
    ncols=5,
    mode="expand",
    borderaxespad=0.0,
)
plt.subplots_adjust(bottom=0.4)
plt.subplots_adjust(left=0.23)
# fig.savefig(f"images/time_spent_{grouping_variable}.png")
# fig.savefig(f"images/time_spent_{grouping_variable}.pdf")
# %%
grouping_variable = "estimator"
fig, axs = plt.subplots(
    1,
    1,
    layout="constrained",
    figsize=(5, 2.2),
)
# fig, axs = plt.subplots(1,2, layout="constrained", sharey=True, figsize=(8,2))
prepare_subplot(df_raw, axs, grouping_variable, "relative")
h, l = axs.get_legend_handles_labels()
labels = [mapping_times[_] for _ in l]
fig.legend(
    h,
    labels,
    bbox_to_anchor=(0, 1.0, 1, 0.3),
    loc="upper left",
    ncols=3,
    mode="expand",
    borderaxespad=0.1,
    labelspacing=0.3,
)

fig.savefig("images/single_time_breakdown.pdf", bbox_inches="tight")
fig.savefig("images/single_time_breakdown.png", bbox_inches="tight")

# %%
grouping_variable = "estimator"
fig, ax_tot = plt.subplots(
    1,
    1,
    layout="constrained",
    figsize=(5, 2.2),
)
# fig, ax_tot = plt.subplots(1,2, layout="constrained", sharey=True, figsize=(8,2))
prepare_subplot(df_raw, ax_tot, grouping_variable, "total")

fig.savefig("images/single_total_time_spent.pdf")
fig.savefig("images/single_total_time_spent.png")


# %%
