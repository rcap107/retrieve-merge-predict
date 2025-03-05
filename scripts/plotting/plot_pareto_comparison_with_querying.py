#%%
%cd ~/bench
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
major_time_locator = FixedLocator([10, 120, 600, 3600, 15000])
gb_locator = FixedLocator([2000, 3000, 4000, 5000, 8000, 10000, 15000, 20000])

#%% RESULTS WITH STARMIE, NO OPEN DATA OR YADL50K
df = pl.read_parquet("results/results_retrieval.parquet")
df = df.with_columns(
    pl.when(pl.col("prediction_metric") < -1)
    .then(-1)
    .otherwise(pl.col("prediction_metric"))
    .alias("y")
).filter(pl.col("estimator") != "nojoin")


query_times_retrieval = pl.read_csv("stats/avg_query_time_for_pareto_plot_retrieval.csv")
query_ram_retrieval = pl.read_csv("stats/dummy_peak_ram.csv")
# query_times_all_datalakes = pl.read_csv("stats/avg_query_time_for_pareto_plot_all_datalakes.csv")

#%%
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

#%%
res = df.group_by(keys).agg(
    pl.mean("y"),
    pl.sum("time_run")/15,
    pl.max("peak_fit"),
).join(query_times_retrieval, on="jd_method").with_columns(
    time_run=pl.col("time_run")+pl.col("time_query")).join(
        query_ram_retrieval, on="jd_method").with_columns(max_ram=pl.max_horizontal("peak_fit", "peak_ram"))

#%%
fig, axs = plt.subplots(
    3,
    3,
    sharey=True,
    layout="constrained",
    figsize=(12, 6),
    gridspec_kw={"wspace": 0.01},
    height_ratios=[0.75,3,3]
)

data = res.to_pandas()

y_var = "y"

map_xlabel = {
    "time_run": "Run time (s)",
    "peak_fit": "Peak RAM (GB)",
    "max_ram": "Peak RAM (GB)",
    "total_time": "Retrieval + Training time (s)",
}

plot_vars = ["time_run", "max_ram"]

for idx_col in range(3):
    ax = axs[0, idx_col]
    ax.axis("off")
    
for _i in range(1,3):
    idx_row = _i
    x_var = plot_vars[idx_row-1]
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
        if idx_row == 1:
            ax.set_xscale("log")

            ax.legend(
                h,
                l,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.7),
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
            ax.xaxis.set_major_locator(gb_locator)
            # ax.xaxis.set_minor_formatter(minor_gb_formatter)

ax_l = axs[1][0]
ax_l.annotate(
    "Pareto\nfrontier",
    (3600, optimal_y),
    (8000, 0.35),
    fontsize="x-small",
    verticalalignment="center",
    horizontalalignment="center",
    #   backgroundcolor="white",
    #   arrowprops=dict(facecolor='black', shrink=0.01, width=1, connectionstyle="arc3,rad=-0.4"),
)

ax_l.annotate(
    "",
    (3000, optimal_y),
    (5000, 0.35),
    fontsize="x-small",
    verticalalignment="center",
    horizontalalignment="center",
    #   backgroundcolor="white",
    arrowprops=dict(
        facecolor="black", shrink=0.01, width=1, connectionstyle="arc3,rad=-0.4"
    ),
)


fig.supylabel("Prediction Performance")

fig.savefig("images/pareto_comparison_with_query.png",bbox_inches="tight" )
fig.savefig("images/pareto_comparison_with_query.pdf", bbox_inches="tight")
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
    pl.sum("time_run")/15,
    pl.max("peak_fit"),
).join(query_times_all_datalakes, on="jd_method").with_columns(time_run=pl.col("time_run")+pl.col("time_query"))


data = res.to_pandas()
fig, axs = plt.subplots(
    3,
    3,
    sharey=True,
    layout="constrained",
    figsize=(12, 6),
    gridspec_kw={"wspace": 0.01},
    height_ratios=[0.75,3,3]
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

for idx_col in range(3):
    ax = axs[0, idx_col]
    ax.axis("off")
    
for idx_row in range(1, 3):
    x_var = plot_vars[idx_row-1]

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
        if idx_row == 1:
            ax.set_xscale("log")

            ax.legend(
                h,
                l,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.7),
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

fig.savefig("images/pareto_comparison_all_datalakes_with_query.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_all_datalakes_with_query.pdf", bbox_inches="tight")

# %%
