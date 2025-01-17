"""
Figure 4: Comparing retrieval methods (including Starmie).
"""

# %%
# %cd ~/store3/retrieve-merge-predict/bench
# %load_ext autoreload
# %autoreload 2
# %%

import matplotlib.pyplot as plt
import polars as pl

from src.utils import constants, plotting
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
df = pl.read_csv("stats/stats_retrieval_plot.csv")
df = df.rename({"index_name": "jd_method"}).filter(
    pl.col("data_lake_version") != "wordnet_vldb_50"
)
# All this garbage is needed to aggregate properly the results
_d = df.with_columns(
    cat=pl.col("jd_method").cast(pl.Categorical).to_physical()
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
fig, axs = plt.subplots(
    1, 2, squeeze=True, layout="constrained", figsize=(7, 3.5), sharey=True
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

# Peak RAM
plotting.prepare_case_subplot(
    axs[0],
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
    axs[1],
    res_time,
    "jd_method",
    plotting_variable="diff_time_retrieval",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)

axs[0].set_title("Peak RAM")
axs[1].set_title("Time difference")

axs[0].annotate(
    f'{res_mem["reference_column"][0]/1000:.0f} GB',
    xy=(1, 4.2),
    xytext=(1.8, 4.1),
    fontsize=14,
    color="tab:blue",
    arrowprops=dict(facecolor="black", width=1, headwidth=5),
)
axs[1].annotate(
    f'{res_time["reference_column"][0]/60:.0f} min',
    xy=(1, 4.2),
    xytext=(1.8, 4.1),
    fontsize=14,
    color="tab:blue",
    arrowprops=dict(facecolor="black", width=1, headwidth=5),
)

fig.savefig("images/boxplot_retrieval_methods.png")
fig.savefig("images/boxplot_retrieval_methods.pdf")


# %%
