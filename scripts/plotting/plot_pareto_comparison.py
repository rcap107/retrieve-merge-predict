#%%
# %cd ~/bench
#%%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src.utils import constants, plotting
from src.utils.logging import read_and_process

sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")


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
major_time_locator = FixedLocator([10, 120, 600, 3600])
gb_locator = FixedLocator([2000, 3000, 4000, 5000, 7000, 10000])

#%% RESULTS WITH STARMIE, NO OPEN DATA OR YADL50K
df = pl.read_parquet("results/results_retrieval.parquet")
df = df.with_columns(
    pl.when(pl.col("prediction_metric") < -1)
    .then(-1)
    .otherwise(pl.col("prediction_metric"))
    .alias("y")
).filter(pl.col("estimator") != "nojoin")
# Setting constants
grouping_variables = {1: "estimator", 2: "chosen_model", 0: "jd_method"}
titles = {0: "Selector", 1: "Prediction Model", 2: "Retrieval method"}
hue_order = {
    "estimator": [
        "full_join",
        "highest_containment",
        "stepwise_greedy_join",
        "best_single_join",
    ],
    "chosen_model": ["catboost", "ridgecv", "resnet", "realmlp"],
    "jd_method": [
        "exact_matching",
        "minhash",
        "minhash_hybrid",
        "starmie",
    ],
    "aggregation": ["first", "mean", "dfs"],
}
palettes = {0: "tab10", 1: "tab10", 2: "tab10"}

keys = ["jd_method", "estimator", "aggregation", "chosen_model"]

res = df.group_by(keys).agg(
    pl.mean("y"),
    pl.mean("time_run"),
    pl.mean("peak_fit"),
)

data = res.to_pandas()
fig, axs = plt.subplots(
    2,
    3,
    sharey=True,
    layout="constrained",
    figsize=(12, 6),
    gridspec_kw={"wspace": 0.01},
)

data = res.to_pandas()
y_var = "y"

map_xlabel = {
    "time_run": "Run time (s)",
    "peak_fit": "Peak RAM (GB)",
    "max_ram": "Peak RAM (GB)",
    "total_time": "Retrieval + Training time (s)",
}

plot_vars = ["time_run", "peak_fit"]

for idx_row in range(2):
    x_var = plot_vars[idx_row]
    for idx_col in range(3):
        grouping_var = grouping_variables[idx_col]
        ax = axs[idx_row][idx_col]
        ax.set_ylim([-0.5, 0.6])
        ax.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")
        if idx_col == 1:
            _xlabel = map_xlabel[x_var]
        else:
            _xlabel = ""
        (h, l), optimal_y = plotting.pareto_frontier_plot(
            data,
            x_var,
            y_var,
            hue_var=grouping_variables[idx_col],
            palette="tab10",
            hue_order=hue_order[grouping_var],
            ax=ax,
            ax_title="",
            ax_xlabel=_xlabel,
        )
        l = [constants.LABEL_MAPPING[grouping_variables[idx_col]][_] for _ in l]

        # first row
        if idx_row == 0:
            ax.set_xscale("log")

            ax.legend(
                h,
                l,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.5),
                title=constants.LABEL_MAPPING["variables"][grouping_var],
                ncols=2,
                mode="expand",
                edgecolor="white",
            )
            ax.xaxis.set_major_formatter(major_time_formatter)
            ax.xaxis.set_major_locator(major_time_locator)
        else:
            ax.get_legend().remove()
            # ax.legend([], [], edgecolor="white")
            ax.set_xscale("log")

            ax.xaxis.set_major_formatter(major_gb_formatter)
            ax.xaxis.set_minor_formatter(minor_gb_formatter)

ax_l = axs[0][0]
ax_l.annotate(
    "Pareto\nfrontier",
    (250, optimal_y),
    (3000, 0.35),
    fontsize="x-small",
    verticalalignment="center",
    horizontalalignment="center",
    #   backgroundcolor="white",
    #   arrowprops=dict(facecolor='black', shrink=0.01, width=1, connectionstyle="arc3,rad=-0.4"),
)

ax_l.annotate(
    "",
    (250, optimal_y),
    (1200, 0.35),
    fontsize="x-small",
    verticalalignment="center",
    horizontalalignment="center",
    #   backgroundcolor="white",
    arrowprops=dict(
        facecolor="black", shrink=0.01, width=1, connectionstyle="arc3,rad=-0.4"
    ),
)


fig.supylabel("Prediction Performance")

fig.savefig("images/pareto_comparison.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison.pdf", bbox_inches="tight")
#%% RESULTS WITH ALL DATA LAKES, NO STARMIE
df = pl.read_parquet("results/results_general.parquet")
df = df.with_columns(
    pl.when(pl.col("prediction_metric") < -1)
    .then(-1)
    .otherwise(pl.col("prediction_metric"))
    .alias("y")
).filter(pl.col("estimator") != "nojoin")
# Setting constants
grouping_variables = {1: "estimator", 2: "chosen_model", 0: "jd_method"}
titles = {0: "Selector", 1: "Prediction Model", 2: "Retrieval method"}
hue_order = {
    "estimator": [
        "full_join",
        "highest_containment",
        "stepwise_greedy_join",
        "best_single_join",
    ],
    "chosen_model": ["catboost", "ridgecv", "resnet", "realmlp"],
    "jd_method": ["exact_matching", "minhash", "minhash_hybrid"],
    "aggregation": ["first", "mean", "dfs"],
}
palettes = {0: "tab10", 1: "tab10", 2: "tab10"}

keys = ["jd_method", "estimator", "aggregation", "chosen_model"]

res = df.group_by(keys).agg(
    pl.mean("y"),
    pl.mean("time_run"),
    pl.mean("peak_fit"),
)

data = res.to_pandas()
fig, axs = plt.subplots(
    2,
    3,
    sharey=True,
    layout="constrained",
    figsize=(12, 6),
    gridspec_kw={"wspace": 0.01},
)

data = res.to_pandas()
y_var = "y"

map_xlabel = {
    "time_run": "Run time (s)",
    "peak_fit": "Peak RAM (GB)",
    "max_ram": "Peak RAM (GB)",
    "total_time": "Retrieval + Training time (s)",
}

plot_vars = ["time_run", "peak_fit"]

for idx_row in range(2):
    x_var = plot_vars[idx_row]
    for idx_col in range(3):
        grouping_var = grouping_variables[idx_col]
        ax = axs[idx_row][idx_col]
        ax.set_ylim([-0.5, 0.6])
        ax.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")
        if idx_col == 1:
            _xlabel = map_xlabel[x_var]
        else:
            _xlabel = ""
        (h, l), _ = plotting.pareto_frontier_plot(
            data,
            x_var,
            y_var,
            hue_var=grouping_variables[idx_col],
            palette="tab10",
            hue_order=hue_order[grouping_var],
            ax=ax,
            ax_title="",
            ax_xlabel=_xlabel,
        )
        l = [constants.LABEL_MAPPING[grouping_variables[idx_col]][_] for _ in l]

        # first row
        if idx_row == 0:
            ax.set_xscale("log")

            ax.legend(
                h,
                l,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.5),
                title=constants.LABEL_MAPPING["variables"][grouping_var],
                ncols=2,
                mode="expand",
                edgecolor="white",
            )
            ax.xaxis.set_major_formatter(major_time_formatter)
            ax.xaxis.set_major_locator(major_time_locator)
        else:
            ax.get_legend().remove()
            # ax.legend([], [], edgecolor="white")
            ax.set_xscale("log")

            ax.xaxis.set_major_formatter(major_gb_formatter)
            ax.xaxis.set_minor_formatter(minor_gb_formatter)

fig.supylabel("Prediction Performance")

fig.savefig("images/pareto_comparison_all_datalakes.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_all_datalakes.pdf", bbox_inches="tight")

# %%
