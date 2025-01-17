"""
Figure: distribution of containment by data lake and retrieval method
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
        ncols=3,
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
    1, 1, squeeze=True, figsize=(6, 4), layout="constrained", sharex=True
)

prepare_containment_plot(fig, ax)

fig.savefig("images/containment-top200-retrieval.pdf", bbox_inches="tight")
fig.savefig("images/containment-top200-retrieval.png", bbox_inches="tight")
# %%
