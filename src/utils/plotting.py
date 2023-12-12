from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Polygon


# Function to add jitter to data
def add_jitter(data, factor=0.1):
    return data + np.random.normal(0, factor, len(data))


def add_subplot(fig, position):
    new_ax = fig.add_subplot(*position, frameon=False)

    new_ax.spines["top"].set_color("none")
    new_ax.spines["bottom"].set_color("none")
    new_ax.spines["left"].set_color("none")
    new_ax.spines["right"].set_color("none")
    new_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    return new_ax


def prepare_clean_labels(labels):
    map_dataset = {
        "company-employees-prepared": "CE",
        "movies-prepared": "CE",
        "presidential-results-prepared": "CE",
    }

    map_agg = {"base_table": "BT", "dedup": "DD", "dfs": "DFS", "none": "NO"}

    map_variant = {"binary": "B", "wordnet": "W"}
    clean_labels = []
    for label in labels:
        dataset, agg, variant = label.split("|")
        # s = f"{map_dataset[dataset]}-{map_agg[agg]}-{map_variant[variant]}"
        s = f"{map_agg[agg]}-{map_variant[variant]}"
        # print(label, s)
        clean_labels.append(s)
    return clean_labels


def prepare_input_data(df: pl.DataFrame, variable_of_interest):
    data = (
        df.with_columns(
            (
                pl.col("source_table")
                + "|"
                + pl.col("aggregation")
                + "|"
                + pl.col("target_dl")
            ).alias("key")
        )
        .select(pl.col("key"), pl.col(variable_of_interest).alias("var"))
        .to_pandas()
    )

    # Dropping one group of base table runs, it could be either wordnet or binary
    data = data.drop(data.loc[data["key"].str.endswith("base_table|wordnet")].index)
    # Converting to categories
    data["key"] = data["key"].astype("category")
    data["index"] = data["key"].cat.codes
    labels = data["key"].unique().categories.to_list()

    formatted_data = []
    for g, group in data.groupby("key"):
        formatted_data.append(group["var"].values)
    return formatted_data, labels


def get_case(label, step):
    """Choose the correct parameters for box filling and hatch depending on the case.

    Args:
        label (str): Label for the given case, has format "dataset|aggregation|yadl variant"
        step (str): Either "color" or "hatch"

    Returns:
        int: An integer value that is used to find the correct index for the task.
    """
    dataset, aggregation, variant = label.split("|")
    if step == "color":
        mapping = {
            "base_table": 0,
            "dedup": 1,
            "dfs": 2,
            "none": 3,
        }
        return mapping[aggregation]
    elif step == "hatch":
        mapping = {"binary": 0, "wordnet": 1, "base_table": 2}
        if aggregation == "base_table":
            return mapping[aggregation]
        else:
            return mapping[variant]
    else:
        raise ValueError(f"Wrong step {step}")


def prepare_boxplot(df, variable_of_interest, ylabel, yscale="lin"):
    formatted_data, labels = prepare_input_data(
        df, variable_of_interest=variable_of_interest
    )
    clean_labels = prepare_clean_labels(labels)

    labels_legend = [
        "Base table",
        "Dedup - Binary",
        "Dedup - Wordnet",
        "DFS - Binary",
        "DFS - Wordnet",
        "None - Binary",
        "None - Wordnet",
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = plt.boxplot(
        formatted_data, notch=False, sym="+", vert=True, whis=1.5, widths=1
    )
    plt.setp(bp["boxes"], color="black")
    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    colors = mpl.colormaps["tab10"].resampled(4).colors
    hatches = ["/", "\\", "."]
    num_boxes = len(formatted_data)

    legend_arguments = []

    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        label = labels[i]
        box = bp["boxes"][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        patch = Polygon(box_coords, facecolor=colors[get_case(label, "color")])
        legend_arguments.append(patch)
        patch.set_hatch(hatches[get_case(label, "hatch")])
        ax1.add_patch(patch)
        # Now draw the median lines back over what we just filled in
        med = bp["medians"][i]
        median_x = []
        median_y = []

    ax1.set_xticklabels(clean_labels, rotation=45)
    ax1.set_ylabel(ylabel)
    if yscale == "log":
        ax1.set_yscale("log")

    # Add a new x axis
    # from https://stackoverflow.com/questions/37934242/hierarchical-axis-labeling-in-matplotlib-python
    ax2 = ax1.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.tick_params("both", length=0, width=0, which="minor")
    ax2.tick_params("both", direction="in", which="major")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Set new ticks on the new axis
    ax2.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([4 / 21, 11 / 21, 18 / 21]))
    ax2.xaxis.set_minor_formatter(
        ticker.FixedFormatter(
            ["Company Employees", "Movies", "US Presidential Elections"]
        )
    )

    # legend
    plt.legend(legend_arguments[:7], labels_legend, loc="upper left")


def base_barplot(
    df: pd.DataFrame,
    x_variable="estimator",
    y_variable="r2score",
    hue_variable="chosen_model",
    col_variable="base_table",
    horizontal=True,
    sharex=False,
    col_wrap=3,
    col_order=None,
):
    if horizontal:
        x = y_variable
        y = x_variable
    else:
        x = x_variable
        y = y_variable
    ax = sns.catplot(
        data=df,
        x=x,
        y=y,
        hue=hue_variable,
        kind="box",
        col=col_variable,
        sharex=sharex,
        col_wrap=col_wrap,
        col_order=col_order,
    )

    return ax


def base_relplot(
    df: pd.DataFrame,
    x_variable="time_run",
    y_variable="r2score",
    hue_variable="chosen_model",
    style_variable="estimator",
    col_variable="base_table",
    sharex=False,
    col_wrap=3,
    col_order=None,
):
    x = x_variable
    y = y_variable

    ax = sns.relplot(
        data=df,
        x=x,
        y=y,
        hue=hue_variable,
        col=col_variable,
        style=style_variable,
        kind="scatter",
        facet_kws={"sharex": sharex, "sharey": True, "subplot_kws": {"xscale": "log"}},
        col_wrap=col_wrap,
        col_order=col_order,
    )
    return ax


def violin_plot_case(
    ax, df, scatterplot_variable, scatterplot_mapping, jitter_factor=0.07
):
    group_keys = ["jd_method", "estimator", "chosen_model", "target_dl", "base_table"]
    data = df["scaled_diff"].to_numpy()
    ax_violin = ax
    ax_violin.axvline(0, alpha=0.4, zorder=0, color="gray")

    parts = ax_violin.violinplot(
        data,
        showmedians=False,
        showmeans=False,
        vert=False,
    )
    for pc in parts["bodies"]:
        pc.set_edgecolor("black")
        pc.set_facecolor("none")
        pc.set_alpha(1)
        pc.set_linewidth(1)
        pc.set_zorder(2.5)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75])
    ax_violin.scatter(
        medians, [1], marker="o", color="white", s=30, zorder=3, edgecolors="black"
    )
    ax_violin.annotate(
        f"{medians:.2f}",
        xy=(medians, -0.2),
        xycoords=(
            "data",
            "axes fraction",
        ),
        size="x-small",
        color="blue",
    )
    ax_violin.axvline(medians, alpha=0.7, zorder=2, color="blue")

    for label in scatterplot_mapping:
        # values = df.filter(pl.col(scatterplot_variable) == label)["scaled_diff"].to_numpy()
        values = (
            df.filter(pl.col(scatterplot_variable) == label)
            .group_by(group_keys)
            .agg(pl.mean("scaled_diff"))["scaled_diff"]
            .to_numpy()
        )
        ax_violin.scatter(
            values,
            add_jitter(np.ones_like(values), jitter_factor),
            color=scatterplot_mapping[label],
            marker="o",
            s=30,
            alpha=0.7,
            label=label,
            zorder=0,
        )
    h, l = ax_violin.get_legend_handles_labels()
    return h, l


def draw_plot(
    cases,
    outer_variable,
    current_results,
    scatterplot_variable="estimator",
    colormap_name="viridis",
):
    group_keys = ["jd_method", "estimator", "chosen_model", "target_dl", "base_table"]
    inner_variables = [_ for _ in cases.keys() if _ != outer_variable]
    n_cols = len(cases[outer_variable])
    n_rows = sum(len(cases[var]) for var in inner_variables)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8, 3),
        sharex=True,
        sharey="row",
        layout="constrained",
    )

    ax_big = add_subplot(fig, (1, 1, 1))

    unique_target = (
        current_results.group_by(pl.col(scatterplot_variable))
        .agg(pl.mean("scaled_diff"))
        .select(pl.col(scatterplot_variable).unique())
        .to_numpy()
        .squeeze()
    )
    colors = plt.colormaps[colormap_name](np.linspace(0, 1, len(unique_target)))
    # colors = plt.cm.viridis(np.linspace(0, 1, len(unique_target)))
    scatterplot_mapping = dict(
        zip(
            unique_target,
            colors,
        )
    )

    for idx_outer_var, c_m in enumerate(cases[outer_variable]):
        print(c_m)
        for idx_inner_var, tg in enumerate(inner_variables):
            new_ax = add_subplot(fig, (2, 1, idx_inner_var + 1))
            new_ax.set_ylabel(tg)
            new_ax.yaxis.set_label_position("left")
            print(tg)
            for idx_plot, c_inner in enumerate(cases[tg]):
                print(c_inner)
                ax = axes[2 * idx_inner_var + idx_plot, idx_outer_var]
                ax.annotate(c_inner + c_m, xy=(0, 0))
                filter_dict = {outer_variable: c_m, tg: c_inner}
                # subset = current_results.filter(**filter_dict).group_by(group_keys).agg(pl.mean("scaled_diff"))
                subset = current_results.filter(**filter_dict)
                h, l = violin_plot_case(
                    ax, subset, scatterplot_variable, scatterplot_mapping
                )
                ax.set_yticks([1], [c_inner])
        axes[0][idx_outer_var].set_title(c_m)
    # for ax, col in zip(axes[0], cols):
    #     ax.set_title(col)
    fig.legend(
        h,
        l,
        loc="outside lower left",
        mode="expand",
        ncols=4,
        markerscale=1,
        borderaxespad=-0.2,
        bbox_to_anchor=(0, -0.1, 1, 0.5),
        scatterpoints=3,
    )
    fig.set_constrained_layout_pads(
        w_pad=5.0 / 72.0, h_pad=4.0 / 72.0, hspace=0.0 / 72.0, wspace=0.0 / 72.0
    )
    fig.suptitle(outer_variable)
    fig.supxlabel("Scaled difference w.r.t. no join")

    # fig.tight_layout()
    # ax_big.legend(h,l)
