"""
Figure 7:
(a) distribution of containment in the query results produced by regression model on each data lake
(b) prediction performance with respect to containment, with regression plot
"""

# %%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

from src.utils import logging, plotting
from src.utils.constants import LABEL_MAPPING, LEGEND_LABELS, ORDER_MAPPING

# %%
sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

STATS_DIR = Path("results/stats")


# %%
def filter_df(df, data_lake):
    target_tables = list(LEGEND_LABELS.keys())
    filtered = df.filter(
        (pl.col("target_dl") == data_lake)
        & (pl.col("base_table").str.contains("depleted"))
    ).with_columns(case=pl.col("base_table").str.split("-").list.first())
    return filtered.filter(pl.col("case").is_in(target_tables))


# %%


### PREPARE REGRESSION
def plot_reg(X, y, ax, label):
    model = LinearRegression()
    model.fit(X, y)
    ax.scatter(
        X + plotting.prepare_jitter(X.shape, offset_value=0, factor=0.0),
        y,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.plot(X, model.predict(X), label=label)
    ax.set_ylabel("Prediction score")
    ax.set_xlabel("Containment")


def prepare_data(df_raw, df_analysis, top_k=200):
    df_agg = (
        df_analysis.filter(pl.col("top_k") == top_k)
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
        )
        .sort("retrieval_method", "table_name")
    )

    res = df_raw.join(
        df_agg,
        left_on=["target_dl", "jd_method", "base_table", "query_column"],
        right_on=[
            "data_lake_version",
            "retrieval_method",
            "table_name",
            "query_column",
        ],
    )

    f = {"jd_method": "exact_matching", "chosen_model": "catboost"}
    r = res.filter(**f)
    X = r.select("avg_containment").to_numpy()
    y = r["y"].to_numpy()
    return X, y


def get_datalake_info(df, data_lake_version):
    df_ = filter_df(df, data_lake=data_lake_version)
    df_ = df_.filter(
        jd_method="exact_matching",
        estimator="stepwise_greedy_join",
        chosen_model="catboost",
    )

    df_analysis = pl.read_csv(
        f"results/stats/analysis_query_results_{data_lake_version}_stats_all.csv"
    )

    return (df_, df_analysis)


# %%
def prepare_regression(fig, ax):
    result_path = "results/overall/overall_first.parquet"

    df_results = pl.read_parquet(result_path)

    current_results = logging.read_and_process(df_results)
    df_overall = current_results.filter(pl.col("estimator") != "nojoin")

    dl_names = [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
        "open_data_us",
    ]

    for name in dl_names:
        df_raw, df_analysis = get_datalake_info(df_overall, name)
        X, y = prepare_data(df_raw, df_analysis)
        plot_reg(X, y, ax, LABEL_MAPPING["target_dl"][name])

    ax.axhline(alpha=0.3)
    ax.set_xlabel("")


# %%
# PREPARE CONTAINMENT PLOT
def prepare_containment_plot(fig, ax):
    dl_names = [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
        "open_data_us",
    ]

    list_df = []

    for name in dl_names:
        _df = pl.read_csv(STATS_DIR / f"analysis_query_results_{name}_stats_all.csv")
        list_df.append(_df)

    df = pl.concat(list_df).filter(pl.col("top_k") == 200)
    order = ORDER_MAPPING["jd_method"]
    sns.boxplot(
        data=df.to_pandas(),
        x="containment",
        y="retrieval_method",
        hue="data_lake_version",
        ax=ax,
        order=order,
        fliersize=2,
    )

    mapping = LABEL_MAPPING["target_dl"]
    h, l = ax.get_legend_handles_labels()
    labels = [mapping[_] for _ in l]
    ax.get_legend().remove()

    fig.legend(
        h,
        labels,
        loc="upper left",
        fontsize=10,
        ncols=5,
        bbox_to_anchor=(0, 1.0, 1, 0.1),
        mode="expand",
    )
    ax.set_yticklabels(
        [LABEL_MAPPING["jd_method"][x.get_text()] for x in ax.get_yticklabels()]
    )
    ax.set_xlabel("")
    # ax.set_xlabel("Containment")
    ax.set_ylabel("")
    # fig.savefig("images/containment-barplot-datalake.pdf", bbox_inches="tight")
    # fig.savefig("images/containment-barplot-datalake.png", bbox_inches="tight")


# %%
fig, ax = plt.subplots(
    1, 2, squeeze=True, figsize=(8, 3), layout="constrained", sharex=True
)

prepare_containment_plot(fig, ax[0])
prepare_regression(fig, ax[1])

fig.savefig("images/containment-regression.pdf", bbox_inches="tight")
fig.savefig("images/containment-regression.png", bbox_inches="tight")


# %%
