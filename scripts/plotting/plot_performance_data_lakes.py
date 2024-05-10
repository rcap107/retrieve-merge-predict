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
fig, axs = plt.subplots(
    1, 2, squeeze=True, figsize=(6, 2), layout="constrained", sharex=True
)

df_ = df_raw.filter(
    (pl.col("jd_method") == "exact_matching")
    & (pl.col("estimator") == "stepwise_greedy_join")
    # & (~pl.col("base_table").str.contains("schools"))
).with_columns(
    r2score=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score"))
)
ax = axs[0]
sns.boxplot(data=df_.to_pandas(), x="r2score", y="target_dl", ax=ax, color="tab:blue")
ax.set_ylabel("")
# ax.set_xlabel("Prediction performance")
ax.set_xlabel("")

mapping = constants.LABEL_MAPPING["target_dl"]

ax.set_yticklabels([mapping[x.get_text()] for x in ax.get_yticklabels()])

#%%
df_ = df_raw.filter(
    (pl.col("target_dl") == "wordnet_vldb_10")
    & (pl.col("estimator") == "stepwise_greedy_join")
    # & (~pl.col("base_table").str.contains("schools"))
).with_columns(
    r2score=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score"))
)


ax = axs[1]
sns.boxplot(data=df_.to_pandas(), x="r2score", y="jd_method", ax=ax, color="tab:blue")
ax.set_ylabel("")
# ax.set_xlabel("Prediction performance")
ax.set_xlabel("")

mapping = {
    "exact_matching": "Exact",
    "minhash": "MinHash",
    "starmie": "Starmie",
    "minhash_hybrid": "H. MinHash",
}

ax.set_yticklabels([mapping[x.get_text()] for x in ax.get_yticklabels()])

# fig.supxlabel("Prediction performance")

# fig.savefig("images/prediction_performance.png")
# fig.savefig("images/prediction_performance.pdf")

# %%
#%%
fig, axs = plt.subplots(
    1, 2, squeeze=True, figsize=(6.5, 2), layout="constrained", sharex=False
)

var_to_plot = f"r2score"

df_ = df_raw.filter(
    (pl.col("jd_method") == "exact_matching")
    & (pl.col("estimator") == "stepwise_greedy_join")
    # & (~pl.col("base_table").str.contains("schools"))
).with_columns(
    r2score=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score"))
)

ax = axs[0]
plotting.prepare_case_subplot(
    ax,
    df=df_,
    grouping_dimension="target_dl",
    scatterplot_dimension=None,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_variable=var_to_plot,
    jitter_factor=0.05,
    scatter_mode="split",
    qle=0,
    xtick_format="linear",
)


df_ = df_raw.filter(
    (pl.col("target_dl") == "wordnet_vldb_10")
    & (pl.col("estimator") == "stepwise_greedy_join")
    # & (~pl.col("base_table").str.contains("schools"))
).with_columns(
    r2score=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score"))
)
ax = axs[1]
plotting.prepare_case_subplot(
    ax,
    df=df_,
    grouping_dimension="jd_method",
    scatterplot_dimension=None,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_variable=var_to_plot,
    jitter_factor=0.05,
    scatter_mode="split",
    qle=0,
    xtick_format="linear",
)

fig.savefig("images/prediction_performance.png")
fig.savefig("images/prediction_performance.pdf")


# %%
