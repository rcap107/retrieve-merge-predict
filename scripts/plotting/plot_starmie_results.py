# %%
# %cd ~/bench
# #%%
# %load_ext autoreload
# %autoreload 2
# %%
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.gridspec import GridSpec

import src.utils.plotting as plotting
from src.utils import constants
from src.utils.logging import read_and_process, read_logs

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)


# %%
def prep_difference(df, column_to_average, result_column):
    prepared_df = _df.with_columns(reference_column=pl.mean(result_column))

    prepared_df = prepared_df.with_columns(
        (pl.col(result_column) / pl.col("reference_column")).alias(
            f"diff_{result_column}"
        )
    )
    return prepared_df


# %%
result_path = "results/overall/overall_first.parquet"

df_results = pl.read_parquet(result_path)
current_results = read_and_process(df_results)

current_results = current_results.filter(pl.col("estimator") != "nojoin")

scatterplot_mapping = plotting.prepare_scatterplot_mapping_case(current_results)
# %%
_d = current_results.filter(
    (pl.col("estimator") != "top_k_full_join")
    & (~pl.col("target_dl").is_in(["wordnet_vldb_50", "open_data_us"]))
)
scatterplot_mapping = plotting.prepare_scatterplot_mapping_case(_d)

# %%
grouping_dimension = "jd_method"
scatter_d = "case"

df_rel_r2 = plotting.get_difference_from_mean(
    _d, column_to_average=grouping_dimension, result_column="r2score"
)
df_rel_time = plotting.get_difference_from_mean(
    _d, column_to_average=grouping_dimension, result_column="time_run", geometric=True
)

formatting_dict = {
    f"diff_{grouping_dimension}_r2score": {"xtick_format": "percentage"},
    f"diff_{grouping_dimension}_time_run": {"xtick_format": "symlog"},
}


# %%
df = pl.read_csv("stats_retrieval_plot.csv")
df = df.rename({"index_name": "jd_method"}).filter(
    pl.col("data_lake_version") != "wordnet_vldb_50"
)
# All this garbage is needed to aggregate properly the results
_d = df.with_columns(
    cat=pl.col("jd_method").cast(pl.Categorical).cast(int)
).with_columns(
    cat=pl.when((pl.col("cat") == 1) | (pl.col("cat") == 2))
    .then(1)
    .otherwise(pl.col("cat"))
)
_d = _d.join(
    _d.group_by(["data_lake_version", "cat"]).agg(
        pl.mean("time_create", "time_save", "time_load")
    ),
    on=["data_lake_version", "cat"],
)

_df = (
    _d.with_columns(
        time_retrieval=pl.when(pl.col("jd_method") != "starmie")
        .then(
            pl.sum_horizontal(
                [
                    "time_create_right",
                    "time_save_right",
                    "time_load_right",
                    "time_query",
                ]
            )
        )
        .otherwise("total_retrieval")
    )
    .group_by(["data_lake_version", "jd_method"])
    .agg(pl.mean("peak_memory"), pl.mean("time_retrieval"))
)
res_mem = prep_difference(_df, "jd_method", "peak_memory")
res_time = prep_difference(_df, "jd_method", "time_retrieval")


# %%
fig, axs = plt.subplots(
    1, 4, squeeze=True, layout="constrained", figsize=(12, 3), sharey=True
)

var_to_plot = f"diff_{grouping_dimension}_r2score"
plotting.prepare_case_subplot(
    axs[0],
    df=df_rel_r2,
    grouping_dimension=grouping_dimension,
    scatterplot_dimension=scatter_d,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_variable=grouping_dimension,
    sorting_method="manual",
    jitter_factor=0.02,
    qle=0.05,
    scatter_mode="split",
    xtick_format=formatting_dict[var_to_plot]["xtick_format"],
    scatterplot_mapping=scatterplot_mapping,
)
axs[0].set_title(
    rf"$R^2$ difference",
)

var_to_plot = f"diff_{grouping_dimension}_time_run"
plotting.prepare_case_subplot(
    axs[1],
    df=df_rel_time,
    grouping_dimension=grouping_dimension,
    scatterplot_dimension=scatter_d,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_variable=grouping_dimension,
    sorting_method="manual",
    jitter_factor=0.03,
    qle=0.05,
    scatter_mode="split",
    xtick_format=formatting_dict[var_to_plot]["xtick_format"],
    scatterplot_mapping=scatterplot_mapping,
)
axs[1].set_title(
    rf"Time difference",
)

locations = [0.01, 1, 2, 3, 5, 10]
labels = [
    r"$0.01x$",
    r"$1x$",
    r"$2x$",
    r"$3x$",
    r"$5x$",
    r"$10x$",
]
symlog_ticks = [locations, labels]

plotting.prepare_case_subplot(
    axs[2],
    res_mem,
    "jd_method",
    plotting_variable="diff_peak_memory",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)

plotting.prepare_case_subplot(
    axs[3],
    res_time,
    "jd_method",
    plotting_variable="diff_time_retrieval",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)

axs[2].set_title("Relative Peak RAM")
axs[3].set_title("Relative retrieval time")

fig.savefig("images/comparison_retrieval_methods.png")
fig.savefig("images/comparison_retrieval_methods.pdf")

# %%
fig, axs = plt.subplots(
    1, 2, squeeze=True, sharey=True, figsize=(10, 2), layout="constrained"
)


# %%
