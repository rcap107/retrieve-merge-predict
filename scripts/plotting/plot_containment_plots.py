# # %%
# %cd ~/bench

# #%%
# %load_ext autoreload
# %autoreload 2

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

import src.utils.plotting as plotting
from src.utils.constants import LABEL_MAPPING

#%%
sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")

DEFAULT_QUERY_RESULT_DIR = Path("results/query_results")


def load_query_result(yadl_version, index_name, tab_name, query_column, top_k):
    query_result_path = "{}__{}__{}__{}.pickle".format(
        yadl_version,
        index_name,
        tab_name,
        query_column,
    )

    with open(Path(DEFAULT_QUERY_RESULT_DIR, query_result_path), "rb") as fp:
        query_result = pickle.load(fp)

    query_result.select_top_k(top_k)
    return query_result


# %%
df_query = pl.read_csv("results/query_logging.txt")
df_index = pl.read_csv("results/index_logging.txt")
df_query.with_columns(
    pl.when(pl.col("index_name") == "exact_matching")
    .then(pl.lit("exact"))
    .otherwise(pl.lit("minhash"))
)
df_minhash = df_index.filter(pl.col("index_name") == "minhash").join(
    df_query, on=["data_lake_version", "index_name"]
)

# %%
df_timings = df_query.join(
    df_index, on=["data_lake_version", "index_name", "base_table", "query_column"]
)
# %%
df_raw = pl.read_parquet("results/overall/open_data_general_first.parquet")
df_raw = df_raw.filter(pl.col("estimator") != "nojoin")
# %%
df = pl.read_csv("analysis_query_results_open_data_us-fixed.csv")
df = df.with_columns(
    (pl.col("cnd_nrows") * pl.col("containment")).alias("matched_rows")
)

# %%
df_agg = (
    df.filter(pl.col("top_k") == 200)
    .group_by(
        [
            "retrieval_method",
            "data_lake_version",
            "table_name",
            "query_column",
            "aggregation",
        ]
    )
    .agg(
        pl.col("containment").mean().alias("avg_containment"),
        pl.col("containment").median().alias("median_containment"),
        pl.col("containment").top_k(30).mean().alias("top_30_avg_containment"),
        pl.col("containment").top_k(30).median().alias("top_30_median_containment"),
        pl.col("cnd_nrows").mean().alias("avg_cnd_nrows"),
        pl.col("cnd_nrows").median().alias("median_cnd_nrows"),
        pl.col("join_time").mean().alias("avg_join_time"),
        pl.col("join_time").median().alias("median_join_time"),
        pl.col("matched_rows").mean().alias("avg_matched_rows"),
        pl.col("matched_rows").median().alias("median_matched_rows"),
    )
    .sort("retrieval_method", "table_name")
)

# %%
order = ["minhash", "minhash_hybrid", "exact_matching"]
fig, ax = plt.subplots(squeeze=True, figsize=(5, 3), layout="constrained")
sns.boxplot(
    data=df.to_pandas(),
    x="containment",
    y="retrieval_method",
    hue="top_k",
    ax=ax,
    order=order,
)
ax.get_legend().remove()
fig.legend(loc="lower left", title="Top-k")
ax.set_yticklabels(
    [LABEL_MAPPING["jd_method"][x.get_text()] for x in ax.get_yticklabels()]
)
ax.set_xlabel("Containment")
ax.set_ylabel("")
# fig.savefig("images/containment-topk.pdf")
# fig.savefig("images/containment-topk.png")


# %%
res = df_raw.join(
    df_agg,
    left_on=["target_dl", "jd_method", "base_table", "query_column"],
    right_on=["data_lake_version", "retrieval_method", "table_name", "query_column"],
)

f = {"jd_method": "exact_matching", "chosen_model": "catboost"}


# %%
from sklearn.linear_model import LinearRegression

r = res.filter(**f)
X = r.select("avg_containment").to_numpy()
y = r["r2score"].to_numpy()

model = LinearRegression()
model.fit(X, y)
fig, ax = plt.subplots(squeeze=True, figsize=(4, 3), layout="constrained")
ax.scatter(
    X + plotting.prepare_jitter(X.shape, offset_value=0, factor=0.0),
    y,
    alpha=0.7,
    edgecolors="k",
    linewidths=0.5,
)
ax.plot(X, model.predict(X), color="k")
ax.set_ylabel("R2 score")
ax.set_xlabel("Containment")

# fig.savefig("images/regplot.pdf")
# fig.savefig("images/regplot.png")

# %%
cmap = mpl.colormaps["Set1"](range(2))
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")

order = (
    res.filter(**f)
    .group_by(["base_table"])
    .agg(pl.median("r2score"))
    .sort("r2score")["base_table"]
    .to_numpy()
)

_d = res.filter(**f).select(["base_table", "r2score"]).melt(id_vars=["base_table"])
sns.boxplot(
    data=_d.to_pandas(), y="base_table", x="value", ax=ax, color=cmap[0], order=order
)
_d = (
    res.filter(**f)
    .select(["base_table", "avg_containment"])
    .melt(id_vars=["base_table"])
)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
sns.pointplot(
    data=_d.to_pandas(),
    y="base_table",
    x="value",
    ax=ax2,
    label="Avg. containment",
    color=cmap[1],
    order=order,
)
# ax.legend(loc="lower left")
ax.set_xlabel("R2 score", color=cmap[0])
# ax2.legend(loc="lower left")
ax2.set_xlabel("Avg containment", color=cmap[1])
ax.set_ylabel(None)
ax.set_yticklabels(
    [LABEL_MAPPING["base_table"][x.get_text()] for x in ax.get_yticklabels()]
)
# fig.savefig("images/r2score_containment_boxplot.pdf")
# fig.savefig("images/r2score_containment_boxplot.png")
# %%
