"""
Figure 4: prediction performance by data lake.
"""

# %%
# %cd ~/bench

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.utils import constants, plotting

# %%
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
result_path = Path("results/overall", "overall_first.parquet")
df_raw = pl.read_parquet(result_path)


# %%
# %%
def prepare_df(df):
    df_ = (
        df.with_columns(case=pl.col("base_table").str.split("-").list.first())
        .filter(
            (pl.col("jd_method") == "exact_matching")
            & (pl.col("estimator") == "stepwise_greedy_join")
            & (pl.col("case").is_in(constants.LEGEND_LABELS.keys()))
        )
        .with_columns(
            y=pl.when(pl.col("auc") > 0)
            .then(pl.col("auc"))
            .otherwise(pl.col("r2score"))
        )
    )
    return df_


# %%

fig, axs = plt.subplots(
    1, 1, squeeze=True, figsize=(5, 2), layout="constrained", sharex=False
)

var_to_plot = f"y"

df_ = prepare_df(df_raw)

# ax = axs[0]
plotting.prepare_case_subplot(
    axs,
    df=df_,
    grouping_dimension="target_dl",
    scatterplot_dimension=None,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_method="manual",
    sorting_variable="target_dl",
    jitter_factor=0.05,
    scatter_mode="split",
    qle=0,
    xtick_format="linear",
)


# df_ = df_raw.filter(
#     (pl.col("target_dl") == "wordnet_vldb_10")
#     & (pl.col("estimator") == "stepwise_greedy_join")
#     # & (~pl.col("base_table").str.contains("schools"))
# ).with_columns(
#     r2score=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score"))
# )
# ax = axs[1]
# plotting.prepare_case_subplot(
#     ax,
#     df=df_,
#     grouping_dimension="jd_method",
#     scatterplot_dimension=None,
#     plotting_variable=var_to_plot,
#     kind="box",
#     sorting_variable=var_to_plot,
#     jitter_factor=0.05,
#     scatter_mode="split",
#     qle=0,
#     xtick_format="linear",
# )

fig.savefig("images/prediction_performance.png")
fig.savefig("images/prediction_performance.pdf")


# %%
