# #%%
# %cd ~/bench
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib import ticker

from src.utils.constants import LABEL_MAPPING

sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="serif")
#%%
df_raw = pl.read_parquet("results/overall/wordnet_general_first.parquet")

#%%
df_query = pl.read_csv("results/query_logging.txt")
df_index = pl.read_csv("results/index_logging.txt")

d_in = (
    df_index.filter(pl.col("data_lake_version") == "wordnet_full")
    .group_by("index_name", "base_table")
    .agg(
        pl.col("time_creation").mean(),
        pl.col("time_save").mean(),
    )
    .group_by("index_name")
    .agg(
        pl.col("time_creation").sum(),
        pl.col("time_save").sum(),
    )
)
d_in = d_in.vstack(
    d_in.filter(pl.col("index_name") == "minhash").with_columns(
        pl.col("index_name").replace("minhash", "minhash_hybrid")
    )
).with_columns(pl.sum_horizontal(cs.numeric()).alias("total_create"))

#%%
grouping_variable = "jd_method"
res_pipeline = (
    df_raw.group_by(grouping_variable)
    .agg(
        # pl.col("time_fit").mean(),
        # pl.col("time_predict").mean(),
        # pl.col("time_run").mean(),
        pl.col("time_prepare").mean(),
        pl.col("time_model_train").mean(),
        pl.col("time_join_train").mean(),
        pl.col("time_model_predict").mean(),
        pl.col("time_join_predict").mean(),
    )
    .with_columns(pl.sum_horizontal(cs.numeric()).alias("total_pipeline"))
)

grouping_variable = "jd_method"
res_pipeline = (
    df_raw.group_by(grouping_variable)
    .agg(
        pl.col("time_run").sum(),
    )
    .with_columns(pl.sum_horizontal(cs.numeric()).alias("total_pipeline"))
)

#%%
fres = (
    res_pipeline.join(
        (
            df_query.filter(pl.col("data_lake_version") == "wordnet_full")
            .drop("query_column", "step", "time_create", "time_save")
            .group_by("index_name")
            .agg(
                pl.col("time_load").sum(),
                pl.col("time_query").sum(),
            )
            .with_columns(pl.sum_horizontal(cs.numeric()).alias("total_query"))
        ),
        left_on="jd_method",
        right_on="index_name",
    )
    .join(
        d_in,
        left_on="jd_method",
        right_on="index_name",
    )
    .select(pl.col("jd_method", "total_pipeline", "total_query", "total_create"))
    .melt(id_vars=["jd_method"])
)
#%%
fig, ax = plt.subplots(
    squeeze=True,
    figsize=(10, 3)
    #    , layout="constrained"
)

sns.barplot(data=fres.to_pandas(), x="value", y="jd_method", hue="variable", ax=ax)
ax.set_ylabel("")
ax.set_xlabel("")
ax.bar_label(ax.containers[0], fontsize=15, fmt="{:.0f}", padding=10)
ax.bar_label(ax.containers[1], fontsize=15, fmt="{:.0f}", padding=10)
ax.bar_label(ax.containers[2], fontsize=15, fmt="{:.0f}", padding=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

major_locator = ticker.FixedLocator(np.arange(0, 18000, 1800))
major_formatter = ticker.FixedFormatter(
    [
        "0",
        "30m",
        "1h",
        "1h30m",
        "2h",
        "2h30m",
        "3h",
        "3h30m",
        "4h",
        "4h30m",
    ]
)
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_major_formatter(major_formatter)

ax.set_yticklabels(
    [LABEL_MAPPING["jd_method"][x.get_text()] for x in ax.get_yticklabels()]
)
h, l = ax.get_legend_handles_labels()
ax.get_legend().remove()

labels = {
    "total_pipeline": "Pipeline",
    "total_query": "Load + Query",
    "total_create": "Index + Persist",
}

fig.legend(
    h,
    [labels[_] for _ in l],
    #    bbox_to_anchor=(0, 0.02, 1, 0.3),
    loc="outside upper left",
    ncols=5,
    mode="expand",
    borderaxespad=0.0,
)

fig.savefig("images/overall_time_spent.png")
fig.savefig("images/overall_time_spent.pdf")

# %%
