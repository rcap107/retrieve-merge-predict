# %%
import os
os.chdir("../..")
# %%
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FuncFormatter,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils import constants


sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %% Define configuration
XTICKS_INSET = [1, 2, 3, 5, 8, 10]

_LEFT_INSET_CONFIG = {
    "inset_loc": "upper left",
    "labelleft": False,
    "labelright": True,
}


_RIGHT_INSET_CONFIG = {
    "inset_loc": "upper right",
    "labelleft": True,
    "labelright": False,
}


INSET_CONFIG = {
    "wordnet_vldb_10": _LEFT_INSET_CONFIG,
    "open_data_us": _LEFT_INSET_CONFIG,
    "wordnet_vldb_50": _RIGHT_INSET_CONFIG,
    "wordnet_full": _RIGHT_INSET_CONFIG,
    "binary_update": _RIGHT_INSET_CONFIG,
}


import numpy as np


def major_gigabyte_formatter(x, pos):
    return f"{x/1e3:.0f} GB"


def minor_gigabyte_formatter(x, pos):
    return f"{x/1e3:.0f}"


def major_time_formatter(x, pos):
    # return f"{x/60:.0f}min"
    if x > 60:
        return f"{x/60:.0f}min"
    else:
        return f"{x:.0f}s"


# Create a FuncFormatter object
major_gb_formatter = FuncFormatter(major_gigabyte_formatter)
minor_gb_formatter = FuncFormatter(minor_gigabyte_formatter)

# Fixed locators
# major_time_locator = FixedLocator([10, 120, 600, 3600])
gb_locator = FixedLocator(np.linspace(2000, 16000, 5))


# %% Read results dataframe
df = pl.read_csv("results/results_topk.csv")
hue_order = df["table"].unique().to_numpy()
# %% Performance metric plot
g = sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="prediction_metric",
    hue="table",
    # row="estimator",
    col="target_dl",
    marker="o",
    kind="line",
    facet_kws={"sharey": True},
    hue_order=hue_order,
    legend=None,
    height=4,
    aspect=1.1
)

for ax in g.axes.flat:
    col_value = ax.get_title().split(" = ")[-1]  # Extract the value after ' = '
    cv = constants.LABEL_MAPPING["target_dl"][col_value]
    ax.set_title(f"{cv}", fontsize=14)
    ax.set_ylabel("Prediction performance")
    ax.set_xlabel("Number of retrieved candidates")


# Extract the colors associated with each hue category
colors = sns.color_palette("tab10", len(hue_order))

# Create the mapping of hue label to color
color_label_mapping = {label: color for label, color in zip(hue_order, colors)}

from matplotlib.lines import Line2D
# Create a custom legend with Line2D objects
legend_handles = [
        Line2D([0], [0], color=color, lw=2, marker='o', markersize=8, label=label) 
    for label, color in color_label_mapping.items()]
legend_labels = [constants.LABEL_MAPPING["base_table"][k] for k in color_label_mapping.keys()]

# Create a new figure for the legend
figlegend = plt.figure(figsize=(3, 1))

# Add the legend to the new figure
plt.legend(legend_handles, legend_labels, loc='center', fontsize=12, ncols=len(legend_handles))
plt.axis("off")

figlegend.savefig("images/legend_topk.png", bbox_inches="tight")
figlegend.savefig("images/legend_topk.pdf", bbox_inches="tight")
g.savefig("images/top_k-prediction_metric.png")
g.savefig("images/top_k-prediction_metric.pdf")



# %% Run time plot
g = sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="time_run",
    hue="table",
    col="target_dl",
    marker="o",
    kind="line",
    facet_kws={"sharey": True},
    hue_order=hue_order,
    legend=None,
        height=4,
    aspect=1.1

)

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(major_time_formatter)
    col_value = ax.get_title().split(" = ")[-1]  # Extract the value after ' = '
    cv = constants.LABEL_MAPPING["target_dl"][col_value]
    ax.set_title(f"{cv}", fontsize=14)
    ax.set_ylabel("Single fold runtime")
    ax.set_xlabel("Number of retrieved candidates")

    _cfg = INSET_CONFIG[col_value]

    ax_inset = inset_axes(ax, width="30%", height="30%", loc=_cfg["inset_loc"])
    sns.lineplot(
        x="top_k",
        y="time_run",
        ax=ax_inset,
        data=df.filter(target_dl=col_value),
        hue="table",
        legend=None,
        marker="o",
        hue_order=hue_order,
    )

    ax_inset.set_xlim(0, 11)
    ax_inset.set_ylim(0, 50)

    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")

    ax_inset.yaxis.set_major_formatter(major_time_formatter)

    ax_inset.tick_params(
        axis="y", labelright=_cfg["labelright"], labelleft=_cfg["labelleft"]
    )

    ax_inset.set_xticks(XTICKS_INSET)
    ax_inset.xaxis.set_minor_locator(
        AutoMinorLocator(1)
    )  # Minor ticks between major ticks on x-axis

    rect = Rectangle(
        (0, 0), 11, 50, linewidth=1, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)

g.savefig("images/top_k-time_run.png")
g.savefig("images/top_k-time_run.pdf")

# %% Peak RAM plot
g = sns.relplot(
    data=df.to_pandas(),
    x="top_k",
    y="peak_fit",
    hue="table",
    col="target_dl",
    marker="o",
    kind="line",
    facet_kws={"sharey": True, "sharex": True},
    hue_order=hue_order,
    legend=None,
        height=4,
    aspect=1.1

)


for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(major_gb_formatter)
    ax.yaxis.set_minor_formatter(minor_gb_formatter)
    col_value = ax.get_title().split(" = ")[-1]  # Extract the value after ' = '
    cv = constants.LABEL_MAPPING["target_dl"][col_value]
    ax.set_title(f"{cv}", fontsize=14)
    ax.set_ylabel("Peak RAM fit")
    ax.set_xlabel("Number of retrieved candidates")

    _cfg = INSET_CONFIG[col_value]

    ax_inset = inset_axes(ax, width="30%", height="30%", loc=_cfg["inset_loc"])
    sns.lineplot(
        x="top_k",
        y="peak_fit",
        ax=ax_inset,
        data=df.filter(target_dl=col_value),
        hue="table",
        legend=None,
        marker="o",
        hue_order=hue_order,
    )

    ax_inset.set_xlim(0, 11)
    ax_inset.set_ylim(1000, 5000)

    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")

    ax_inset.yaxis.set_major_formatter(major_gb_formatter)
    ax_inset.yaxis.set_minor_formatter(minor_gb_formatter)

    ax_inset.tick_params(
        axis="y", labelright=_cfg["labelright"], labelleft=_cfg["labelleft"]
    )

    ax_inset.set_xticks(XTICKS_INSET)
    ax_inset.xaxis.set_minor_locator(
        AutoMinorLocator(1)
    )  # Minor ticks between major ticks on x-axis

    rect = Rectangle(
        (0, 1000),
        11,
        5000,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

g.savefig("images/top_k-peak_fit.png")
g.savefig("images/top_k-peak_fit.pdf")


# %% Pivot tables
def prepare_pivot_table(df, variable):
    return (
        df.sort("top_k")
        .pivot(
            on="top_k",
            index="target_dl",
            values=variable,
            aggregate_function="mean",
        )
        .with_columns(pl.col("target_dl").replace(constants.LABEL_MAPPING["target_dl"]))
        .with_columns(cs.numeric().round(3))
    )


_1 = prepare_pivot_table(df, "prediction_metric")
_2 = prepare_pivot_table(df, "time_run")
_3 = prepare_pivot_table(df, "peak_fit")

