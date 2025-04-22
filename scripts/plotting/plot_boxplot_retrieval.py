"""
Figure 25: Comparing retrieval methods (including Starmie).
"""

# %%
import os

os.chdir("../..")
import matplotlib.pyplot as plt
import polars as pl

from src.utils import plotting


# %%
def prep_difference(df, result_column):
    prepared_df = df.with_columns(
        reference_column=pl.col("jd_method") == "exact_matching"
    )

    prepared_df = prepared_df.with_columns(
        (pl.col(result_column) / pl.col("reference_column")).alias(
            f"diff_{result_column}"
        )
    )
    return prepared_df

def preprocess_data():
    df = pl.read_csv("stats/stats_retrieval_plot.csv")
    df = df.rename({"index_name": "jd_method"}).filter(
        pl.col("data_lake_version") != "wordnet_vldb_50"
    )
    # Aggregating results
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
    df_prep = _df.join(
        _df.filter(pl.col("jd_method") == "exact_matching"), on="data_lake_version"
    ).with_columns(
        diff_ram=pl.col("peak_memory") / pl.col("peak_memory_right"),
        diff_time=pl.col("time_retrieval") / pl.col("time_retrieval_right"),
    )

    return df_prep
#%%
df_prep = preprocess_data()
# %%
fig, axs = plt.subplots(
    1, 2, squeeze=True, layout="constrained", figsize=(7, 3.5), sharey=True
)


locations = [1, 2, 5, 20, 100]
labels = [
    # r"$0.01x$",
    r"$1x$",
    r"$2x$",
    # r"$3x$",
    r"$5x$",
    # r"$10x$",
    r"$20x$",
    r"$100x$",
]
symlog_ticks = [locations, labels]

# Peak RAM
plotting.prepare_case_subplot(
    axs[0],
    df_prep,
    "jd_method",
    plotting_variable="diff_ram",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)
# Retrieval time difference
plotting.prepare_case_subplot(
    axs[1],
    df_prep,
    "jd_method",
    plotting_variable="diff_time",
    sorting_method="manual",
    sorting_variable="jd_method",
    xtick_format="symlog",
    qle=0.02,
    symlog_ticks=symlog_ticks,
)

axs[0].set_title("Peak RAM")
axs[1].set_title("Time difference")


annotations = df_prep.group_by("jd_method").agg(
    pl.mean("peak_memory"), pl.mean("time_retrieval")
)
annot_ram = annotations.filter(jd_method="exact_matching")["peak_memory"].item()
annot_time = annotations.filter(jd_method="exact_matching")["time_retrieval"].item()
axs[0].annotate(
    f"{annot_ram:.0f} MB",
    xy=(1, 4.2),
    xytext=(1.8, 4.1),
    fontsize=14,
    color="tab:blue",
    arrowprops=dict(facecolor="black", width=1, headwidth=5),
)
axs[1].annotate(
    f"{annot_time/60:.0f} min",
    xy=(1, 4.2),
    xytext=(1.8, 4.1),
    fontsize=14,
    color="tab:blue",
    arrowprops=dict(facecolor="black", width=1, headwidth=5),
)

fig.savefig("images/boxplot_retrieval_methods.png")
fig.savefig("images/boxplot_retrieval_methods.pdf")


# %%
