"""
Figure 8:
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

STATS_DIR = Path("stats")


# %%
def filter_df(df, data_lake):
    # Filtering only the results we are interested in.
    target_tables = list(LEGEND_LABELS.keys())
    filtered = df.filter(
        (pl.col("target_dl") == data_lake)
        # & (pl.col("base_table").str.contains("depleted"))
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
    # Preparation function to have consistent results for every DL.
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
    ).with_columns(table_name=pl.col("table_name").str.split("-").list.first())

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
    y = r["prediction_metric"].to_numpy()
    return X, y


def get_datalake_info(df, data_lake_version):
    df_ = filter_df(df, data_lake=data_lake_version)
    df_ = df_.filter(
        jd_method="exact_matching",
        estimator="stepwise_greedy_join",
        chosen_model="catboost",
    )

    df_analysis = pl.read_csv(
        STATS_DIR / f"analysis_query_results_{data_lake_version}_stats_all.csv"
    )

    return (df_, df_analysis)


# %%
def prepare_regression(fig, ax):
    """Given the figure and axes object, prepare ther egression plot. The function
    is expecting to find the data in the given path.
    """
    # Reading and preparing the results
    result_path = "results/temp_results_general.parquet"
    df_results = pl.read_parquet(result_path)
    # current_results = logging.read_and_process(df_results)
    df_overall = df_results.filter(pl.col("estimator") != "nojoin")

    # Comment out the names of the data lakes that should not be printed
    dl_names = [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
        "open_data_us",
    ]

    # Prepare the plot for each data lake.
    for name in dl_names:
        df_raw, df_analysis = get_datalake_info(df_overall, name)
        X, y = prepare_data(df_raw, df_analysis)
        plot_reg(X, y, ax, LABEL_MAPPING["target_dl"][name])

    # Adding a horizontal line for the value of 0
    ax.axhline(alpha=0.3)
    ax.set_xlabel("Containment")


# %%
# PREPARE CONTAINMENT PLOT
def prepare_containment_plot(fig, ax):
    """Given the figure and the axes object to write on, prepare the plot that compares the containment across different
    data lakes.
    """

    # Comment out the names of the data lakes that should not be printed
    dl_names = [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
        "open_data_us",
    ]

    list_df = []

    # These files must already be present.
    for name in dl_names:
        _df = pl.read_csv(STATS_DIR / f"analysis_query_results_{name}_stats_all.csv")
        list_df.append(_df)

    # Assuming that top_k=200 is already there for all data lakes.
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

    # Getting the mapping name and creating the legend.
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
        bbox_to_anchor=(0, 1.0, 1, 0.15),
        mode="expand",
    )
    ax.set_yticklabels(
        [LABEL_MAPPING["jd_method"][x.get_text()] for x in ax.get_yticklabels()]
    )
    ax.set_xlabel("")
    ax.set_xlabel("Containment")
    ax.set_ylabel("")


# %%
fig, ax = plt.subplots(
    1, 2, squeeze=True, figsize=(10, 2.5), layout="constrained", sharex=True
)

prepare_containment_plot(fig, ax[0])
prepare_regression(fig, ax[1])

fig.savefig("images/containment-regression.pdf", bbox_inches="tight")
fig.savefig("images/containment-regression.png", bbox_inches="tight")
# %%
