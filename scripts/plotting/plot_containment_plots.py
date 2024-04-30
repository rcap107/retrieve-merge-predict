"""
Figure 5(b): prediction performance with respect to containment, with regression plot.
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

import src.utils.plotting as plotting
from src.utils.constants import LABEL_MAPPING

# %%
sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

STATS_DIR = Path("results/stats")


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


def prepare_data(df_raw, df_analysis):
    df_agg = (
        df_analysis.filter(pl.col("top_k") == 30)
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
    y = r["r2score"].to_numpy()
    return X, y


def get_open_data():
    df_raw = pl.read_parquet("results/overall/open_data_us-first.parquet")
    df_raw = df_raw.filter(
        (pl.col("estimator") != "nojoin")
        & (~pl.col("base_table").str.contains("schools"))
        & (pl.col("base_table").str.contains("depleted"))
    ).with_columns(
        base_table=pl.col("base_table")
        .str.split("-")
        .list.gather([0, 2])
        .list.join("-")
    )

    df_analysis = pl.read_csv(
        "results/stats/analysis_query_results_open_data_us_stats_all.csv"
    )
    df_analysis = df_analysis.with_columns(
        (pl.col("cnd_nrows") * pl.col("containment")).alias("matched_rows")
    ).with_columns(
        containment=(
            pl.when(pl.col("containment") > 1)
            .then(pl.col("containment") / pl.col("src_nrows"))
            .otherwise(pl.col("containment"))
        )
    )
    return df_raw, df_analysis


def get_wordnet_10():
    df_raw = pl.read_parquet("results/overall/wordnet-10k_first.parquet")
    df_raw = df_raw.filter(pl.col("estimator") != "nojoin")

    df_analysis = pl.read_csv(
        "results/stats/analysis_query_results_wordnet_vldb_10_stats_all.csv"
    )
    df_analysis = df_analysis.with_columns(
        (pl.col("cnd_nrows") * pl.col("containment")).alias("matched_rows")
    ).with_columns(
        containment=(
            pl.when(pl.col("containment") > 1)
            .then(pl.col("containment") / pl.col("src_nrows"))
            .otherwise(pl.col("containment"))
        )
    )
    return df_raw, df_analysis


def get_wordnet_base():
    df_raw = pl.read_parquet("results/overall/old-versions_first.parquet")
    df_raw = df_raw.filter(
        (pl.col("estimator") != "nojoin") & (pl.col("target_dl") == "wordnet_full")
    )

    df_analysis = pl.read_csv(
        "results/stats/analysis_query_results_wordnet_full_stats_all.csv"
    )
    df_analysis = df_analysis.with_columns(
        (pl.col("cnd_nrows") * pl.col("containment")).alias("matched_rows")
    )
    return df_raw, df_analysis


# %%
def prepare_regression(fig, ax):
    df_raw_od, df_analysis_od = get_open_data()
    df_raw_10, df_analysis_10 = get_wordnet_10()
    df_raw_wn, df_analysis_wn = get_wordnet_base()

    # fig, ax = plt.subplots(squeeze=True, figsize=(4, 3), layout="constrained")
    X, y = prepare_data(df_raw_wn, df_analysis_wn)
    plot_reg(X, y, ax, "YADL Base")
    X, y = prepare_data(df_raw_od, df_analysis_od)
    plot_reg(X, y, ax, "Open Data")
    X, y = prepare_data(df_raw_10, df_analysis_10)
    plot_reg(X, y, ax, "YADL 10k")
    # h, labels = ax.get_legend_handles_labels()
    # fig.legend(
    #     h,
    #     labels,
    #     loc="upper left",
    #     fontsize=10,
    #     ncols=3,
    #     bbox_to_anchor=(0, 1.0, 1, 0.1),
    #     mode="expand",
    # )
    ax.axhline(alpha=0.3)


# fig.savefig("images/regplot.pdf", bbox_inches="tight")
# fig.savefig("images/regplot.png", bbox_inches="tight")


# %%
# PREPARE CONTAINMENT PLOT
def prepare_containment_plot(fig, ax):
    df_opendata = pl.read_csv(
        STATS_DIR / "analysis_query_results_open_data_us_stats_all.csv"
    )
    df_wordnet = pl.read_csv(
        STATS_DIR / "analysis_query_results_wordnet_full_stats_all.csv"
    )
    df_vldb_10 = pl.read_csv(
        STATS_DIR / "analysis_query_results_wordnet_vldb_10_stats_all.csv"
    )
    df = pl.concat(
        [
            df_wordnet,
            df_opendata,
            df_vldb_10,
        ]
    ).filter(pl.col("top_k") == 30)
    df = df.with_columns(
        containment=(
            pl.when(pl.col("containment") > 1)
            .then(pl.col("containment") / pl.col("src_nrows"))
            .otherwise(pl.col("containment"))
        )
    )
    order = ["minhash", "minhash_hybrid", "exact_matching", "starmie"]
    # fig, ax = plt.subplots(squeeze=True, figsize=(4, 3), layout="constrained")
    sns.boxplot(
        data=df.to_pandas(),
        x="containment",
        y="retrieval_method",
        hue="data_lake_version",
        ax=ax,
        order=order,
    )

    mapping = {
        "wordnet_full": "YADL Comb.",
        "open_data_us": "Open Data US",
        "wordnet_vldb_10": "YADL 10k",
    }

    h, l = ax.get_legend_handles_labels()
    labels = [mapping[_] for _ in l]
    ax.get_legend().remove()

    fig.legend(
        h,
        labels,
        loc="upper left",
        fontsize=10,
        ncols=3,
        bbox_to_anchor=(0, 1.0, 1, 0.1),
        mode="expand",
    )
    ax.set_yticklabels(
        [LABEL_MAPPING["jd_method"][x.get_text()] for x in ax.get_yticklabels()]
    )
    ax.set_xlabel("")
    # ax.set_xlabel("Containment")
    ax.set_ylabel("")
    fig.savefig("images/containment-barplot-datalake.pdf", bbox_inches="tight")
    fig.savefig("images/containment-barplot-datalake.png", bbox_inches="tight")


# %%
fig, ax = plt.subplots(
    2, 1, squeeze=True, figsize=(5, 6), layout="constrained", sharex=True
)

prepare_containment_plot(fig, ax[0])
prepare_regression(fig, ax[1])
# %%
