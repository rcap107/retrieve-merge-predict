from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Polygon


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
    sns.catplot(
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

    sns.relplot(
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
