# %%
import os

os.chdir("../..")
# %%
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src.utils import constants, plotting
from src.utils.logging import prepare_full_time_run
from matplotlib.lines import Line2D


sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")


def major_gigabyte_formatter(x, pos):
    return f"{x/1e3:.0f}GB"


def major_time_formatter(x, pos):
    if x > 60:
        return f"{x/60:.0f}m"


# Create a FuncFormatter object
major_ram_formatter = FuncFormatter(major_gigabyte_formatter)

# Fixed locators
# Values are tweaked manually for readibility
major_time_locator = FixedLocator([10, 120, 600, 3600, 15000])
gb_locator_starmie = FixedLocator(
    [2000, 4000, 8000, 15000, 145_000]
)
gb_locator_general = FixedLocator([1500, 5000, 7500, 10000, 15000, 20000])
# Aggregation needs lower RAM
gb_locator_aggr = FixedLocator([3000,  5000, 7000, 12000, 15000])

formatter_mapping = {
    "total_runtime": (major_time_formatter, major_time_locator),
    "peak_ram": (major_ram_formatter, gb_locator_general),
}
formatter_mapping_aggr = {
    "total_runtime": (major_time_formatter, major_time_locator),
    "peak_ram": (major_ram_formatter, gb_locator_aggr),
}
# %%
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
palettes = {0: "tab10", 1: "tab10", 2: "tab10", 3: "tab10"}

keys = ["jd_method", "estimator", "aggregation", "chosen_model"]

y_var = "y"

map_xlabel = {
    "time_run": "Run time (s)",
    "peak_fit": "Peak RAM (GB)",
    "max_ram": "Peak RAM (GB)",
    "total_runtime": "Retrieval + Training time (s)",
}

legend_handles = {}
legend_labels = {}


def aggregate_data(df):
    res = prepare_full_time_run(df)
    res = (
        res.group_by(keys)
        .agg(
            pl.mean("y"),
            pl.mean("time_run"),
            pl.mean("time_query"),
            pl.mean("peak_ram"),
        )
        .with_columns(total_runtime=pl.col("time_query") + pl.col("time_run"))
    )
    return res


def get_results_all_datalakes():
    # RESULTS WITH OPEN DATA AND YADL50
    df = pl.read_parquet("results/results_general.parquet")
    # Clamping negative results to avoid breaking the scale
    df = df.with_columns(
        pl.when(pl.col("prediction_metric") < -1)
        .then(-1)
        .otherwise(pl.col("prediction_metric"))
        .alias("y")
    ).filter(pl.col("estimator") != "nojoin")
    res = aggregate_data(df)
    data = res.to_pandas()
    return data


def get_results_starmie():
    # RESULTS WITH STARMIE, NO OPEN DATA OR YADL50K
    df = pl.read_parquet("results/results_retrieval.parquet")
    # Clamping negative results to avoid breaking the scale
    df = df.with_columns(
        pl.when(pl.col("prediction_metric") < -1)
        .then(-1)
        .otherwise(pl.col("prediction_metric"))
        .alias("y")
    ).filter(pl.col("estimator") != "nojoin")
    res = aggregate_data(df)
    data = res.to_pandas()
    return data


def get_results_aggregation():
    # Aggregation results
    df = pl.read_parquet("results/results_aggregation.parquet")
    # Clamping negative results to avoid breaking the scale
    df = df.with_columns(
        pl.when(pl.col("prediction_metric") < -1)
        .then(-1)
        .otherwise(pl.col("prediction_metric"))
        .alias("y")
    ).filter(pl.col("estimator") != "nojoin")
    res = aggregate_data(df)
    data = res.to_pandas()
    return data


def prepare_pareto_plot_general(data, x_var, annotate=False):
    fig = plt.figure(figsize=(10, 1.5))

    _formatter, _locator = formatter_mapping[x_var]
    subfigs = fig.subfigures(1, 3)
    for idx_col in range(3):
        grouping_var = grouping_variables[idx_col]

        subfig = subfigs[idx_col]
        ax_ = subfig.subplots(1, 1, squeeze=True)
        
        if x_var == "peak_ram":
            mm = data[x_var].min()
            mm = mm - 0.1*mm
            ax_.set_xlim([mm, 15000])
        
        ax_.set_ylim([-0.5, 0.6])
        ax_.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")

        if idx_col > 0:
            ax_.tick_params(labelleft=False)

        (h, l), optimal_y = plotting.pareto_frontier_plot(
            data,
            x_var,
            y_var,
            hue_var=grouping_variables[idx_col],
            palette="tab10",
            hue_order=hue_order[grouping_var],
            ax=ax_,
            ax_title="",
            ax_xlabel="",
        )
        l = [constants.LABEL_MAPPING[grouping_variables[idx_col]][_] for _ in l]

        ax_.set_xscale("log")

        ax_.xaxis.set_major_formatter(_formatter)
        ax_.xaxis.set_major_locator(_locator)
        ax_.xaxis.set_minor_locator(NullLocator())

        legend_handles[grouping_var] = h
        legend_labels[grouping_var] = l

        ax_.get_legend().remove()

    # Adding annotation to the first plot
    if annotate:
        subfig = subfigs[0]
        ax_l = subfig.get_axes()[0]

        ax_l.annotate(
            "Pareto\nfrontier",
            (3600, optimal_y),
            (8000, -0.30),
            fontsize="x-small",
            verticalalignment="center",
            horizontalalignment="center",
        )

        ax_l.annotate(
            "",
            (3000, optimal_y),
            (5000, -0.30),
            fontsize="x-small",
            verticalalignment="center",
            horizontalalignment="center",
            arrowprops=dict(
                facecolor="black", shrink=0.01, width=1, connectionstyle="arc3,rad=-0.4"
            ),
        )

    return fig


def prepare_pareto_plot_ram(data):
    # No annotation, but there needs to be a split axis
    fig = plt.figure(figsize=(10, 1.5))

    # Creating one subfigure for each variable
    subfigs = fig.subfigures(1, 3, wspace=0.05)
    x_var = "peak_ram"
    for idx_col in range(0, 3):
        this_fig = subfigs[idx_col]
        grouping_var = grouping_variables[idx_col]

        # The subfigure is split in two for the broken axis
        axs = this_fig.subplots(1, 2, sharey=True, width_ratios=[3, 1])
        this_fig.subplots_adjust(wspace=0.05)

        # Specifying lims for the axes
        ax_ = axs[0]
        ax_right = axs[1]
        ax_.set_ylim([-0.5, 0.6])

        ax_.set_xlim([2000, 21000])
        ax_right.set_xlim([140000, 150000])

        ax_.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")
        ax_right.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")

        # Removing tick labels from all figures except the leftmost
        if idx_col > 0:
            ax_.tick_params(labelleft=False)

        (h, l), optimal_y = plotting.pareto_frontier_plot(
            data,
            x_var,
            y_var,
            hue_var=grouping_variables[idx_col],
            palette="tab10",
            hue_order=hue_order[grouping_var],
            ax=ax_,
            ax_title="",
            ax_xlabel="",
        )

        (h, l), optimal_y = plotting.pareto_frontier_plot(
            data,
            x_var,
            y_var,
            hue_var=grouping_variables[idx_col],
            palette="tab10",
            hue_order=hue_order[grouping_var],
            ax=ax_right,
            ax_title="",
            ax_xlabel="",
        )

        l = [constants.LABEL_MAPPING[grouping_variables[idx_col]][_] for _ in l]

        # Removing legend
        ax_.get_legend().remove()
        ax_right.get_legend().remove()

        # Prettying up the broken axes
        # Hiding the axis edges to make it look seamless
        ax_.spines.right.set_visible(False)
        ax_right.spines.left.set_visible(False)
        ax_right.tick_params(labelleft=False)  # don't put tick labels at the top
        ax_right.tick_params(left=False)  # Hide the ticks on the left side of ax_right
        ax_right.tick_params(
            labelleft=False
        )  # Hide the tick labels on the left side of ax_right

        # Prepare the markers
        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-d, -1), (d, 1)],
            markersize=12,
            linestyle="none",
            color="k",
            mec="k",
            mew=1,
            clip_on=False,
        )
        ax_.plot([1, 1], [1, 0], transform=ax_.transAxes, **kwargs)
        ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kwargs)

        # Axis formatting
        ax_.set_xscale("log")

        ax_.xaxis.set_major_formatter(major_ram_formatter)
        ax_.xaxis.set_major_locator(gb_locator_starmie)
        ax_right.xaxis.set_major_formatter(major_ram_formatter)
        ax_right.xaxis.set_major_locator(gb_locator_starmie)

        ax_.xaxis.set_minor_locator(NullLocator())

    return fig


def prepare_pareto_plot_aggregation(data, x_var):
    # Used for aggregation figures in the appendix
    fig, axs = plt.subplots(
        1,
        4,
        squeeze=True,
        sharey=True,
        sharex=True,
        figsize=(14, 2.5),
        # gridspec_kw={"hspace": 0.4},
        layout="constrained",
    )

    y_var = "y"
    hue_order = {
        "estimator": [
            "highest_containment",
            "best_single_join",
        ],
        "chosen_model": ["catboost", "ridgecv", "resnet", "realmlp"],
        "jd_method": [
            "exact_matching",
            "minhash",
            "minhash_hybrid",
        ],
        "aggregation": ["first", "mean", "dfs"],
    }

    _formatter, _locator = formatter_mapping_aggr[x_var]
    grouping_variables = ["jd_method", "estimator", "aggregation", "chosen_model"]
    for idx_col in range(4):
        grouping_var = grouping_variables[idx_col]
        ax = axs[idx_col]
        if x_var =="total_runtime":
            ax.set_xscale("log")    
        (h, l), _ = plotting.pareto_frontier_plot(
            data,
            x_var,
            y_var,
            hue_var=grouping_var,
            palette="tab10",
            hue_order=hue_order[grouping_var],
            ax=ax,
            ax_title="",
            ax_xlabel="",
        )
        l = [constants.LABEL_MAPPING[grouping_variables[idx_col]][_] for _ in l]

        legend_handles[grouping_var] = h
        legend_labels[grouping_var] = l
        # Removing legend
        ax.get_legend().remove()
        ax.xaxis.set_major_formatter(_formatter)
        ax.xaxis.set_major_locator(_locator)
    return fig


def prepare_pareto_plot_aggregation_single(data):
    # This function is for a single figure in the main body.
    fig, ax = plt.subplots(1, 1, squeeze=True, sharey=True, sharex=True, figsize=(5, 3))

    variable = "time_run"
    y_var = "y"

    group_variable = "aggregation"
    ax.set_xscale("log")
    (h, l), _ = plotting.pareto_frontier_plot(
        data,
        variable,
        y_var,
        hue_var=group_variable,
        palette="tab10",
        hue_order=hue_order[group_variable],
        ax=ax,
        ax_title="",
        ax_xlabel="",
    )
    l = [constants.LABEL_MAPPING["aggregation"][_] for _ in l]

    ax.legend(
        h, l, title="Aggregation", ncols=1, edgecolor="white", title_fontsize="small"
    )
    ax.xaxis.set_major_formatter(major_time_formatter)
    ax.xaxis.set_major_locator(major_time_locator)

    ax.set_ylabel("Prediction Performance")
    ax.set_xlabel("Time run (s)")

    return fig


def prepare_legend(with_starmie=False):
    fig = plt.figure(figsize=(10, 1))

    subfigs = fig.subfigures(1, 3)

    for idx in range(3):
        grouping_var = grouping_variables[idx]

        # Extract the colors associated with each hue category
        colors = sns.color_palette("tab10", len(hue_order[grouping_var]))

        # Create the mapping of hue label to color
        color_label_mapping = {
            label: color for label, color in zip(hue_order[grouping_var], colors)
        }

        # Create a custom legend with Line2D objects
        _legend_handles = [
            Line2D([0], [0], color=color, lw=2, marker="o", markersize=8, label=label)
            for label, color in color_label_mapping.items()
        ]
        _legend_labels = legend_labels[grouping_var]
        print(_legend_labels)

        # Patching the case where Starmie is not being plotted
        if (not with_starmie) and grouping_var == "jd_method":
            _legend_handles = _legend_handles[:-1]
            _legend_labels = _legend_labels[:-1]

        # Create a new figure for the legend
        figlegend = subfigs[idx]
        # Add the legend to the new figure
        ax_ = figlegend.subplots(1, 1, squeeze=True)
        ax_.legend(
            _legend_handles,
            _legend_labels,
            loc="center",
            fontsize=12,
            ncols=2,
            title=constants.LABEL_MAPPING["variables"][grouping_var],
            frameon=False,
        )
        ax_.axis("off")

    return fig


def prepare_legend_aggregation():
    fig = plt.figure(figsize=(14, 1))

    hue_order = {
        "estimator": [
            "highest_containment",
            "best_single_join",
        ],
        "chosen_model": ["catboost", "ridgecv", "resnet", "realmlp"],
        "jd_method": [
            "exact_matching",
            "minhash",
            "minhash_hybrid",
        ],
        "aggregation": ["first", "mean", "dfs"],
    }
    subfigs = fig.subfigures(1, 4)

    grouping_variables = {
        2: "aggregation",
        0: "jd_method",
        1: "estimator",
        3: "chosen_model",
    }
    for idx in range(4):
        grouping_var = grouping_variables[idx]

        # Extract the colors associated with each hue category
        colors = sns.color_palette("tab10", len(hue_order[grouping_var]))

        # Create the mapping of hue label to color
        color_label_mapping = {
            label: color for label, color in zip(hue_order[grouping_var], colors)
        }

        # Create a custom legend with Line2D objects
        _legend_handles = [
            Line2D([0], [0], color=color, lw=2, marker="o", markersize=8, label=label)
            for label, color in color_label_mapping.items()
        ]
        _legend_labels = legend_labels[grouping_var]
        print(_legend_labels)

        # Create a new figure for the legend
        figlegend = subfigs[idx]
        # Add the legend to the new figure
        ax_ = figlegend.subplots(1, 1, squeeze=True)
        ax_.legend(
            _legend_handles,
            _legend_labels,
            loc="center",
            fontsize=12,
            ncols=2,
            title=constants.LABEL_MAPPING["variables"][grouping_var],
            frameon=False,
        )
        ax_.axis("off")

    return fig


# %%
#    ###############
# Starmie
#    ###############
data = get_results_starmie()
fig = prepare_pareto_plot_general(data, x_var="total_runtime", annotate=True)
fig.savefig("images/pareto_comparison_with_query_time.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_with_query_time.pdf", bbox_inches="tight")

fig = prepare_pareto_plot_ram(data)
fig.savefig("images/pareto_comparison_with_query_ram.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_with_query_ram.pdf", bbox_inches="tight")

fig = prepare_legend(with_starmie=True)
fig.savefig("images/pareto_comparison_with_query_legend.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_with_query_legend.pdf", bbox_inches="tight")


# %%    ###############
# All data lakes, no Starmie

data = get_results_all_datalakes()
fig = prepare_pareto_plot_general(data, x_var="total_runtime")
fig.savefig("images/pareto_comparison_general_with_query_time.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_general_with_query_time.pdf", bbox_inches="tight")

fig = prepare_pareto_plot_general(data, x_var="peak_ram")
fig.savefig("images/pareto_comparison_general_with_query_ram.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_general_with_query_ram.pdf", bbox_inches="tight")

fig = prepare_legend()
fig.savefig(
    "images/pareto_comparison_general_with_query_legend.png", bbox_inches="tight"
)
fig.savefig(
    "images/pareto_comparison_general_with_query_legend.pdf", bbox_inches="tight"
)

# %% Full aggregation Pareto
data = get_results_aggregation()
fig = prepare_pareto_plot_aggregation(data, "total_runtime")
fig.savefig("images/pareto_comparison_aggr_with_query_time.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_aggr_with_query_time.pdf", bbox_inches="tight")
fig = prepare_pareto_plot_aggregation(data, "peak_ram")
fig.savefig("images/pareto_comparison_aggr_with_query_ram.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_aggr_with_query_ram.pdf", bbox_inches="tight")
fig = prepare_legend_aggregation()
fig.savefig("images/pareto_comparison_aggr_with_query_legend.png", bbox_inches="tight")
fig.savefig("images/pareto_comparison_aggr_with_query_legend.pdf", bbox_inches="tight")
# %% Single aggregation Pareto
data = get_results_aggregation()
fig = prepare_pareto_plot_aggregation_single(data)
fig.savefig("images/pareto_aggregation_time_single.png", bbox_inches="tight")
fig.savefig("images/pareto_aggregation_time_single.pdf", bbox_inches="tight")
# %%
