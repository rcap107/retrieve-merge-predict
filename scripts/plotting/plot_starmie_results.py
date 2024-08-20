"""
Figure 4: Comparing retrieval methods (including Starmie).
"""

# %%
# %cd ~/bench
# %%
# %load_ext autoreload
# %autoreload 2
# %%

import matplotlib.pyplot as plt
import polars as pl

from src.utils import plotting, constants
from src.utils.logging import read_and_process


# %%
def prep_difference(df, result_column):
    prepared_df = df.with_columns(reference_column=pl.mean(result_column))

    prepared_df = prepared_df.with_columns(
        (pl.col(result_column) / pl.col("reference_column")).alias(
            f"diff_{result_column}"
        )
    )
    return prepared_df


# %%
result_path = "stats/overall/overall_first.parquet"

df_results = pl.read_parquet(result_path)
current_results = read_and_process(df_results)
others = [col for col in current_results.columns if col not in constants.GROUPING_KEYS + ["case"]]
current_results = current_results.group_by(constants.GROUPING_KEYS + ["case"]).agg(
pl.mean(others)
)

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
    _d, column_to_average=grouping_dimension, result_column="y"
)
df_rel_time = plotting.get_difference_from_mean(
    _d, column_to_average=grouping_dimension, result_column="time_run", geometric=True
)

formatting_dict = {
    f"diff_{grouping_dimension}_y": {"xtick_format": "percentage"},
    f"diff_{grouping_dimension}_time_run": {"xtick_format": "symlog"},
}


# %%
df = pl.read_csv("stats/stats_retrieval_plot.csv")
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
# %%
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
res_mem = prep_difference(_df, "peak_memory")
res_time = prep_difference(_df, "time_retrieval")


# %%
fig = plt.figure(layout="constrained", figsize=(12, 3))
subfigs = fig.subfigures(
    1,
    2,
    #  wspace=0.07
)

# Left subfig for retrieval (it's the first step)
subfigs[0].set_facecolor("bisque")
subfigs[0].suptitle("Retrieval")

# Right subfig for prediction pipeline (it's the second step)
subfigs[1].set_facecolor("lightcyan")
subfigs[1].suptitle("Prediction pipeline")

axs_left = subfigs[0].subplots(1, 2, sharey=True)
axs_right = subfigs[1].subplots(1, 2, sharey=True)

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

# Peak RAM
plotting.prepare_case_subplot(
    axs_left[0],
    res_mem,
    "jd_method",
    plotting_variable="diff_peak_memory",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)
# Retrieval time difference
plotting.prepare_case_subplot(
    axs_left[1],
    res_time,
    "jd_method",
    plotting_variable="diff_time_retrieval",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)

axs_left[0].set_title("Peak RAM")
axs_left[1].set_title("Time difference")

print("scatter: ", scatter_d)
# Prediction performance
var_to_plot = f"diff_{grouping_dimension}_y"
plotting.prepare_case_subplot(
    axs_right[0],
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
axs_right[0].set_title(
    rf"$R^2$ difference",
)

# Time prediction pipeline
var_to_plot = f"diff_{grouping_dimension}_time_run"
plotting.prepare_case_subplot(
    axs_right[1],
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
axs_right[1].set_title(
    rf"Time difference",
)

# Remove the labels from the right axis
axs_right[0].yaxis.set_major_locator(plt.NullLocator())

fig.savefig("images/comparison_retrieval_methods.png")
fig.savefig("images/comparison_retrieval_methods.pdf")


# %%
