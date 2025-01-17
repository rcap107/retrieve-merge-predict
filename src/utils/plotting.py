from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import ListedColormap

from src.utils import constants

plt.style.use("seaborn-v0_8-talk")


# fold vs fold difference
def diff_fold_vs_fold(df: pl.DataFrame, column_to_average: str):
    this_key = [_ for _ in constants.GROUPING_KEYS if not _ in [column_to_average]]

    r = df.join(df.filter(**constants.REFERENCE_CONFIG), on=this_key).with_columns(
        (pl.col("prediction_metric") - pl.col("prediction_metric_right")).alias(
            f"diff_{column_to_average}_prediction_metric"
        ),
        (pl.col("time_run") / pl.col("time_run_right")).alias(
            f"diff_{column_to_average}_time_run"
        ),
    )
    return r


def diff_fold_vs_reference_median(df: pl.DataFrame, column_to_average: str):
    this_key = [_ for _ in constants.GROUPING_KEYS if not _ in [column_to_average]]
    this_groupby = [
        _ for _ in constants.GROUPING_KEYS if not _ in [column_to_average, "fold_id"]
    ]

    df_reference = df.filter(**constants.REFERENCE_CONFIG)
    _temp = df_reference.group_by(this_groupby).agg(
        median_prediction=pl.median("prediction_metric"),
        median_time=pl.median("time_run"),
    )
    df_reference = df_reference.join(_temp, on=this_groupby)
    r = df.join(df_reference, on=this_key).with_columns(
        (pl.col("prediction_metric") - pl.col("median_prediction")).alias(
            f"diff_{column_to_average}_prediction_metric"
        ),
        (pl.col("time_run") / pl.col("median_time")).alias(
            f"diff_{column_to_average}_time_run"
        ),
    )
    return r


def diff_mean_vs_mean(df: pl.DataFrame, column_to_average: str):
    grouping_nofold = [_ for _ in constants.GROUPING_KEYS if _ != "fold_id"]
    this_groupby = [
        _ for _ in constants.GROUPING_KEYS if not _ in [column_to_average, "fold_id"]
    ]

    _df = df.group_by(grouping_nofold).agg(
        pl.mean("prediction_metric"), pl.mean("time_run")
    )
    r = _df.join(
        _df.filter(**constants.REFERENCE_CONFIG), on=this_groupby
    ).with_columns(
        (pl.col("prediction_metric") - pl.col("prediction_metric_right")).alias(
            f"diff_{column_to_average}_prediction_metric"
        ),
        (pl.col("time_run") / pl.col("time_run_right")).alias(
            f"diff_{column_to_average}_time_run"
        ),
    )
    return r


def get_difference_from_reference(
    df: pl.DataFrame,
    column_to_average: str,
    result_column: str,
    geometric: bool = False,
):
    if column_to_average not in df:
        raise ValueError
    if result_column not in df:
        raise ValueError

    df_reference = (
        df.filter(**constants.REFERENCE_CONFIG)
        .group_by(
            constants.GROUPING_KEYS
            # [_ for _ in constants.GROUPING_KEYS if _ != "fold_id"]
        )
        .agg(pl.mean(result_column))
    )

    this_groupby = [
        _ for _ in constants.GROUPING_KEYS if not _ in [column_to_average, "fold_id"]
    ]

    prepared_df = df.join(df_reference, on=this_groupby)

    if geometric:
        prepared_df = prepared_df.with_columns(
            (pl.col(result_column) / pl.col(f"{result_column}_right")).alias(
                f"diff_{column_to_average}_{result_column}"
            )
        )
    else:
        prepared_df = prepared_df.with_columns(
            (pl.col(result_column) - pl.col(f"{result_column}_right")).alias(
                f"diff_{column_to_average}_{result_column}"
            )
        )
    return prepared_df.drop(cs.ends_with("_right"))


def get_difference_from_mean(
    df: pl.DataFrame,
    column_to_average: str,
    result_column: str,
    scaled: bool = False,
    geometric: bool = False,
    force_split: bool = False,
):
    """This function takes as input the results dataframe, the result column (measuring either R2 score, AUC or runtime),
    as well as a single column to average on, and builds a new column that includes the difference between the value in
    the result column and the average value for the given `column_to_average`.

    The difference can be the absolute difference between two values, or a `geometric` difference to see "how many times"
    the given value is larger or smaller than the average (this is mostly relevant when working with the execution time).

    Args:
        df (pl.DataFrame): Dataframe that contains the results.
        column_to_average (str): Column of interest (one of the experimental variables).
        result_column (str): The result column to find the average of.
        scaled (bool, optional): If true, scale the results by the absolute maximum difference (useful to represent
                                    values as a percentage of the maximum difference). Defaults to False.
        geometric (bool, optional): If true, the difference is found by executing the ratio between each sample and the
                                    average. Defaults to False.
        force_split (bool, optional): Plotting flag to not use the "relative" difference in case of binary values.
                                    Defaults to False.

    Returns:
        pl.DataFrame: A copy of the original dataframe, with the new column.
    """
    assert column_to_average in df.columns
    assert result_column in df.columns

    this_groupby = [_ for _ in constants.GROUPING_KEYS if _ != column_to_average]

    n_unique = df.select(pl.col(column_to_average).n_unique()).item()
    # The comparison is one "one method against another", but "what's the best among 3+ methods".
    if n_unique > 2 or force_split:
        prepared_df = df.join(
            df.group_by(this_groupby).agg(
                pl.mean(result_column).alias("reference_column")
            ),
            on=this_groupby,
        )

    else:
        # One-vs-one case: find which is the best method in median
        best_method = (
            df.group_by(column_to_average)
            .agg(pl.median(result_column))
            .top_k(1, by=result_column)[column_to_average]
            .item()
        )

        # Find the performance of the best method, and compare it with the performance of the other method
        prepared_df = (
            df.filter(pl.col(column_to_average) == best_method)
            .join(df.filter(pl.col(column_to_average) != best_method), on=this_groupby)
            .filter(pl.col(column_to_average) != pl.col(column_to_average + "_right"))
            .rename({result_column + "_right": "reference_column"})
        )

    # "How many times is method X faster/slower than the reference?"
    if geometric:
        prepared_df = prepared_df.with_columns(
            (pl.col(result_column) / pl.col("reference_column")).alias(
                f"diff_{column_to_average}_{result_column}"
            )
        )
    else:
        # By what % is method X better/worse than the reference?
        prepared_df = prepared_df.with_columns(
            (pl.col(result_column) - pl.col("reference_column")).alias(
                f"diff_{column_to_average}_{result_column}"
            )
        )

    if scaled:
        prepared_df = prepared_df.with_columns(
            prepared_df.with_columns(
                pl.col(f"diff_{column_to_average}_{result_column}")
                / pl.col(f"diff_{column_to_average}_{result_column}").abs().max()
            )
        )

    return prepared_df.drop(cs.ends_with("_right"))


def prepare_jitter(shape, offset_value, factor=0.1):
    data = np.ones(shape) * offset_value
    return data + np.random.normal(0, factor, data.shape)


def prepare_scatterplot_mapping_case(df: pl.DataFrame):
    assert "target_dl" in df.columns
    assert "case" in df.columns

    def get_cmap(cmap_name, n_colors):
        c = plt.colormaps[cmap_name].resampled(n_colors)
        cmap = ListedColormap(colors=c(range(n_colors)))
        return cmap

    maps = []
    for gdx, group in df.sort("target_dl", "case").group_by(
        ["target_dl"], maintain_order=True
    ):
        cases = group.select(pl.col("case").unique()).sort("case")["case"].to_numpy()
        cmap = get_cmap(constants.COLORMAP_DATALAKE_MAPPING[gdx[0]], len(cases) + 1)
        maps += list(zip(cases, cmap.colors))
    scatterplot_mapping = dict(maps)
    return scatterplot_mapping


def prepare_scatterplot_mapping_general(
    df, scatterplot_dimension, plotting_variable, colormap_name="viridis"
):
    # Prepare the labels for the scatter plot and the corresponding colors.
    scatterplot_labels = (
        df.group_by(pl.col(scatterplot_dimension))
        .agg(pl.mean(plotting_variable))
        .sort(plotting_variable)
        .select(pl.col(scatterplot_dimension).unique())
        .to_numpy()
        .squeeze()
    )
    colors = plt.colormaps[colormap_name].resampled(len(scatterplot_labels)).colors
    scatterplot_mapping = dict(
        zip(
            scatterplot_labels,
            colors,
        )
    )
    return scatterplot_mapping


def format_xaxis(ax, case, limits, xmax=1, symlog_ticks=None):
    """Formatting x-axes for the pair plots. Values are assigned manually based
    on "what looks best".

    Args:
        ax (_type_): Axis object to fix.
        case (_type_): Type of axis (percentage, log, symlog, linear).
        limits (_type_): Limits of the axis (for spacing purposes).
        xmax (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if case == "percentage":
        if max(np.abs(limits)) < 0.10:
            major_locator = ticker.MultipleLocator(0.05)
            minor_locator = ticker.MultipleLocator(0.01)
        elif 0.10 <= max(np.abs(limits)) < 0.30:
            major_locator = ticker.MultipleLocator(0.10)
            minor_locator = ticker.MultipleLocator(0.05)
        elif 0.30 <= max(np.abs(limits)):
            major_locator = ticker.MultipleLocator(0.20)
            minor_locator = ticker.MultipleLocator(0.05)
        else:
            raise ValueError
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=xmax, decimals=0))
        # ax.xaxis.set_minor_formatter(ticker.PercentFormatter(xmax=xmax, decimals=0))
        # ax.set_xlim((-0.4, 0.4))
    elif case == "log":
        ax.set_xscale("log", base=2)
        # major_locator = ticker.LogLocator(
        #     base=2,
        # )

        # major_locator =

        ax.xaxis.set_major_locator(major_locator)
        minor_locator = ticker.LogLocator(base=2)
        major_formatter = ticker.FixedFormatter(
            [
                r"$0.5x$",
                r"$1x$",
                r"$1.5x$",
                r"$2x$",
                r"$3x$",
                r"$4x$",
                r"$5x$",
            ]
        )
        ax.xaxis.set_major_formatter(major_formatter)
        # ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=2, labelOnlyBase=False))
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

    elif case == "linear":
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

    elif case == "symlog":
        ax.set_xscale("symlog", base=2)
        major_locator = ticker.SymmetricalLogLocator(
            base=2,
            linthresh=0.025,
        )

        if symlog_ticks is None:
            locations = [0.5, 1, 1.5, 2, 3, 5, 10, 20, 40]
            labels = [
                r"$0.5x$",
                r"$1x$",
                r"$1.5x$",
                r"$2x$",
                r"$3x$",
                r"$5x$",
                r"$10x$",
                r"$20x$",
                r"$40x$",
            ]
        else:
            locations = symlog_ticks[0]
            labels = symlog_ticks[1]

        major_locator = ticker.FixedLocator(locations)
        major_formatter = ticker.FixedFormatter(labels)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(_custom_formatter))
    return ax


def base_barplot(
    df: pd.DataFrame,
    categorical_variable: str = "estimator",
    result_variable: str = "y",
    hue_variable: str = "chosen_model",
    col_variable: str = "base_table",
    horizontal: bool = True,
    sharex: bool = False,
    col_wrap: int = 3,
    col_order: list | None = None,
):
    """Simple utility function for preparing a summary barplot at the end of an experiment. The function takes the
    results dataframe as input and outputs a boxplot that logs the `result_variable` for the given run.

    The `categorical_variable` is used to split results, a boxplot will be drawn for each distinct category found in
    the column.

    The `result_variable` is a numerical variable that is used to build the boxplot proper.

    Args:
        df (pd.DataFrame): Dataframe that contains the results of the experiment.
        x_variable (str, optional): Categorical variable to split results over. Defaults to "estimator".
        y_variable (str, optional): Numerical variable to plot. Defaults to "y".
        hue_variable (str | None, optional): If provided, color the bars by this variable. Defaults to "chosen_model".
        col_variable (str, optional): If provided, create a new subplot for each distinct value in this variable. Defaults to "base_table".
        horizontal (bool, optional): If True, plot horizontal barplots. Defaults to True.
        sharex (bool, optional): If true, the x-axis will be shared across subplots. Defaults to False.
        col_wrap (int, optional): Number of subplots to be drawn in each row. Defaults to 3.
        col_order (list | None, optional): Order of labels in the column. Defaults to None.

    Returns:
        _type_: _description_
    """

    if horizontal:
        x = result_variable
        y = categorical_variable
    else:
        x = categorical_variable
        y = result_variable
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
    y_variable="y",
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


def prepare_case_subplot(
    ax,
    df: pl.DataFrame,
    grouping_dimension: str,
    scatterplot_dimension: str = None,
    plotting_variable: str = None,
    kind: str = "box",
    xtick_format: str = "percentage",
    scatter_mode: str = "overlapping",
    jitter_factor: float = 0.05,
    scatterplot_mapping=None,
    colormap_name: str = "viridis",
    scatterplot_marker_size: float = 6,
    box_width: float = 0.9,
    xmax: float = 1,
    sorting_method: str = "prediction",
    sorting_variable: str = "prediction_metric",
    qle: float = 0.05,
    symlog_ticks=None,
):
    if sorting_method == "prediction":
        if sorting_variable == plotting_variable:
            data = (
                df.select(grouping_dimension, sorting_variable)
                .group_by(grouping_dimension)
                .agg(pl.all(), pl.median(sorting_variable).alias("sort_col"))
                .sort("sort_col")
                .to_dict()
            )
        else:
            data = (
                df.select(grouping_dimension, plotting_variable, sorting_variable)
                .group_by(grouping_dimension)
                .agg(pl.all(), pl.median(sorting_variable).alias("sort_col"))
                .sort("sort_col")
                .select(grouping_dimension, plotting_variable)
                .to_dict()
            )

    elif sorting_method == "manual":
        order_mapping = constants.ORDER_MAPPING[sorting_variable]

        data = (
            df.with_columns(
                pl.col(grouping_dimension)
                .map_elements(order_mapping.index)
                .alias("order")
            )
            .sort("order", descending=True)
            .select(grouping_dimension, plotting_variable)
            .group_by(grouping_dimension, maintain_order=True)
            .agg(pl.all())
            .to_dict()
        )
    # Prepare the plotting data sorting by `sorting_variable` (r2score by default to have  consistency over axes)

    # Axis limits are set manually based on the quantile. The value of qle depends on the variable
    limits = (
        df.select(
            pl.col(plotting_variable).quantile(qle).alias("min"),
            pl.col(plotting_variable).quantile(1 - qle).alias("max"),
        )
        .transpose()
        .to_numpy()
        .squeeze()
    )

    # This is the reference vertical line.
    ref_vline = 1 if xtick_format in ["log", "symlog"] else 0
    ax.axvline(ref_vline, alpha=0.6, zorder=0, color="tab:blue", linestyle="--")

    medianprops = dict(linewidth=2, color="red", zorder=3)
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    if scatterplot_dimension is None:
        boxprops = dict(facecolor="white", linewidth=2, zorder=2)
    else:
        boxprops = dict(facecolor="white", linewidth=2, zorder=2)
    bp = ax.boxplot(
        data[plotting_variable],
        showfliers=False,
        vert=False,
        widths=box_width,
        medianprops=medianprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        # zorder=3,
        patch_artist=True,
    )

    # Find the medians of the values (to be added to the plot)
    df_medians = df.group_by(grouping_dimension).agg(pl.col(plotting_variable).median())
    median_d = dict(zip(*df_medians.to_dict().values()))

    facecolors = ["grey", "white"]

    if scatterplot_dimension is not None:
        # If no scatterplot mapping is provided, build one based on the scatterplot dimension
        if scatterplot_mapping is None:
            scatterplot_mapping = prepare_scatterplot_mapping_general(
                df,
                scatterplot_dimension,
                plotting_variable,
                colormap_name=colormap_name,
            )

        for _i, _d in enumerate(data[plotting_variable]):
            # Add an horizontal span to split the different cases by color
            ax.axhspan(
                _i + 0.5, _i + 1.5, facecolor=facecolors[_i % 2], zorder=0, alpha=0.10
            )

            # Add a white dot to highlight the median
            median = np.median(_d)
            ax.scatter(
                median,
                [_i + 1],
                marker="o",
                color="white",
                s=30,
                zorder=3.5,
                edgecolors="black",
            )

            # Split the points in the scatter plot to reduce overlap
            if scatter_mode == "split":
                offset = (
                    np.linspace(
                        -box_width / 2
                        + jitter_factor,  # needed to stay within the bounds of the plot
                        box_width / 2 - jitter_factor,
                        len(scatterplot_mapping),
                    )
                ) * 0.8
            elif scatter_mode == "overlapping":
                offset = np.zeros(len(scatterplot_mapping))

            # Plot only one set of points at a time
            for _, label in enumerate(scatterplot_mapping):
                this_label = constants.LABEL_MAPPING[scatterplot_dimension][label]

                # Needed to avoid replicating labels
                if _i > 0:
                    this_label = "_" + str(this_label)

                # Selecting only the points that belong to the case in the current iteration
                filter_dict = {
                    scatterplot_dimension: label,
                    grouping_dimension: data[grouping_dimension][_i],
                }
                values = df.filter(**filter_dict)[plotting_variable].to_numpy()

                # Plotting the points after adding jitter.
                ax.scatter(
                    values,
                    _i
                    + 1
                    + prepare_jitter(
                        len(values), offset_value=offset[_], factor=jitter_factor
                    ),
                    color=scatterplot_mapping[label],
                    marker="o",
                    s=scatterplot_marker_size,
                    alpha=0.3,
                    label=this_label,
                    zorder=2.5,
                )
    h, l = ax.get_legend_handles_labels()
    if len(data[plotting_variable]) == 1:
        # Only one value for the plotting variable (e.g., diff. between ML models)
        ax.set_yticks(
            [1], [constants.LABEL_MAPPING["single_label"][grouping_dimension]]
        )
    else:
        # Assign the proper label based on what's in constants.py to the ticks
        ax.set_yticks(
            range(1, len(data[grouping_dimension]) + 1),
            [
                constants.LABEL_MAPPING[grouping_dimension][_l]
                for e, _l in enumerate(data[grouping_dimension])
            ],
        )
    # Adding an annotation with the median of the value for each box in the plot.
    for _i, _l in enumerate(data[grouping_dimension], start=1):
        annot_value = median_d[_l]
        if xtick_format == "percentage":
            annot_value *= 100
            annot_string = f"{annot_value:+.2f}%"
        elif xtick_format == "symlog":
            annot_string = f"{annot_value:+.2f}x"
        else:
            annot_string = f"{annot_value:+.2f}"
        ax.annotate(
            annot_string,
            xy=(limits[1], _i),
            xytext=(limits[1] + 0.03 * limits[1], _i - 0.1),
            # xytext=(limits[1] + 0.03 * limits[1], _i - 0.2),
            xycoords="data",
            textcoords="data",
            fontsize=12,
        )

    ax.set_xlim(limits)
    ax = format_xaxis(ax, xtick_format, limits, xmax=xmax, symlog_ticks=symlog_ticks)

    xlim = ax.get_xlim()

    # Adding a red vspan to highlight the "worse" part of the plot.
    if xtick_format in ["symlog", "log"]:
        ax.axvspan(1, xlim[1], zorder=0, alpha=0.05, color="red")
    else:
        ax.axvspan(xlim[0], 0, zorder=0, alpha=0.05, color="red")

    return h, l


def draw_pair_comparison(
    df: pl.DataFrame,
    grouping_dimension: str,
    scatterplot_dimension: str,
    scatter_mode: str = "overlapping",
    colormap_name: str = "viridis",
    savefig: bool = False,
    savefig_type: list | str = "png",
    savefig_name: str | None = None,
    savefig_tag: str = "",
    case: str = "dep",
    jitter_factor: float = 0.03,
    qle: float = 0.05,
    add_titles: bool = True,
    subplot_titles=None,
    sorting_variable: str = "prediction_metric",
    sorting_method: str = "prediction",
    result_column: str = "prediction_metric",
    figsize=(10, 4),
    axes=None,
    figure=None,
):
    """This function is used to prepare the paired plots used for the paper and other material.

    It will prepare two subplots side-by-side; the first plot presents the relative difference in prediction performance
    from the "reference method" (i.e., the method provided as `grouping_dimension`).

    The performance of the "reference method" itself is measured by finding the maximum difference between all methods
    and that measured for the NoJoin case. This difference is then used as "reference point" against which all other methods
    are compared. The median difference from the reference is also shown on the right of each plot.

    Args:
        df (pl.DataFrame): Dataframe that contains the results.
        grouping_dimension (str): Which variable should be used to find the "reference" method (data lake, ml method, retrieval method etc.)
        scatterplot_dimension (str): Which variable should be use to plot the scatterplot. Normally, this should be "case".
        scatter_mode (str, optional): How to plot the scatterplot. Either "split" or "overlapping". Defaults to "overlapping".
        colormap_name (str, optional): If provided, it will override the default colormap values. Defaults to "viridis".
        savefig (bool, optional): If True, save the figure. Defaults to False.
        savefig_type (list | str, optional): List of extensions to be used for saving the figure. Defaults to "png".
        savefig_name (str | None, optional): If given, save the figure using this name. Defaults to None.
        savefig_tag (str, optional): Additional tags to be added to the fig name to distinguish cases. Defaults to "".
        case (str, optional): Either "full" or "dep", used as tag for the figure. Defaults to "dep".
        jitter_factor (float, optional): Jitter added ot the scatter plot to separate the dots. Defaults to 0.03.
        qle (float, optional): Quantile value to filter out outliers in the scatterplot and have better tick spacing. Defaults to 0.05.
        add_titles (bool, optional): If True, add titles to the subplots. Defaults to True.
        subplot_titles (_type_, optional): _description_. Defaults to None.
        sorting_variable (str, optional): Variable that should be used for sorting the methods. Defaults to "y".
        sorting_method (str, optional): Either "prediction" or "manual"; if "manual", a fixed order will be used. Defaults to "prediction".
        figsize (tuple, optional): Tuple that defines the size of the resulting figure. Defaults to (10, 4).
        axes (_type_, optional): Optional parameter to pass the axes from an external function. Defaults to None.
    """

    # df_rel = diff_mean_vs_mean(df, grouping_dimension)
    # df_rel = diff_fold_vs_reference_median(df, grouping_dimension)
    df_rel = diff_fold_vs_fold(df, grouping_dimension)

    if scatterplot_dimension == "case":
        scatterplot_mapping = prepare_scatterplot_mapping_case(df)
    else:
        scatterplot_mapping = prepare_scatterplot_mapping_general(
            df, scatterplot_dimension, "scaled_diff", colormap_name
        )

    if axes is None:
        fig, axes = plt.subplot_mosaic(
            [[0, 1]],
            layout="constrained",
            figsize=figsize,
            sharey=True,
            width_ratios=(2, 2),
        )
    else:
        fig = figure
    plotting_variables = [
        # f"prediction_metric",
        f"diff_{grouping_dimension}_prediction_metric",
        # f"time_run",
        f"diff_{grouping_dimension}_time_run",
    ]

    formatting_dict = {
        # f"prediction_metric": {"xtick_format": "percentage"},
        # f"time_run": {"xtick_format": "symlog"},
        f"diff_{grouping_dimension}_prediction_metric": {"xtick_format": "percentage"},
        f"diff_{grouping_dimension}_time_run": {"xtick_format": "symlog"},
    }
    if subplot_titles is None:
        subplot_titles = [
            rf"Performance difference",
            rf"Time difference",
        ]

    if scatter_mode is None:
        scatter_mode = "split" if len(scatterplot_mapping) > 2 else "overlapping"

    for idx, var_to_plot in enumerate(plotting_variables[::], start=0):
        ax = axes[idx]
        # ax.grid(which="both", axis="x", alpha=0.3)
        h, l = prepare_case_subplot(
            ax,
            df_rel,
            grouping_dimension,
            scatterplot_dimension,
            plotting_variable=var_to_plot,
            scatterplot_mapping=scatterplot_mapping,
            scatter_mode=scatter_mode,
            xtick_format=formatting_dict[var_to_plot]["xtick_format"],
            kind="box",
            jitter_factor=jitter_factor,
            qle=qle,
            sorting_variable=sorting_variable,
            sorting_method=sorting_method,
        )
        if add_titles:
            axes[idx].set_title(subplot_titles[idx], loc="left")

    # fig.set_constrained_layout_pads(
    #     w_pad=5.0 / 72.0, h_pad=4.0 / 72.0, hspace=0.0 / 72.0, wspace=5.0 / 72.0
    # )

    if savefig:
        if isinstance(savefig_type, str):
            savefig_type = [savefig_type]
        if savefig_name is not None:
            print(savefig_name)
            fig.savefig(Path("images", savefig_name))
        else:
            for ext in savefig_type:
                fname = f"{case}_pair_{grouping_dimension}_{scatterplot_dimension}"
                if savefig_tag:
                    fname += f"_{savefig_tag}"
                fname += f".{ext}"
                print(fname)
                fig.savefig(Path("images", fname))


def prepare_grouped_stacked_barplot_time(
    df: pl.DataFrame, first_var: str, second_var: str
):
    assert first_var in df.columns
    assert second_var in df.columns

    df_prepare = (
        df.group_by([first_var, second_var])
        .agg(
            # pl.col("time_run").mean(),
            pl.col("time_prepare").mean(),
            pl.col("time_model_train").mean(),
            pl.col("time_join_train").mean(),
            pl.col("time_model_predict").mean(),
            pl.col("time_join_predict").mean(),
        )
        .melt(id_vars=[first_var, second_var])
        .sort(first_var, second_var)
    )

    # Map an index to each distinct value in first_var and second_var
    fv_c = first_var + "_c"
    sv_c = second_var + "_c"
    treated = (
        df_prepare.join(
            df_prepare.select(pl.col(first_var).unique()).with_row_count(name=fv_c),
            on=first_var,
        ).join(
            df_prepare.select(pl.col(second_var).unique()).with_row_count(name=sv_c),
            on=second_var,
        )
    ).sort(fv_c, sv_c)

    # Build the cumsum and bottom columns required for the stacked barplot
    to_concat = []
    for _, gr in treated.group_by((first_var, second_var)):
        new_g = (
            gr.sort("variable")
            .with_columns(pl.col("value").cumsum().alias("csum"))
            .with_columns(
                pl.col("csum").alias("bottom").shift(1).fill_null(0),
            )
        )
        to_concat.append(new_g)
    df_c = pl.concat(to_concat)

    # Find the unique values in the first variable
    unique_fv = df_c[fv_c].unique()
    # Find the unique values in the second variable
    unique_sv = df_c[sv_c].unique()

    # Find the number of unique values in the first variable
    n_unique_fv = len(unique_fv)
    # Find the number of unique values in the second variable
    n_unique_sv = len(unique_sv)

    # Prepare a palette with 5 colors (one for each step of the pipeline)
    cmap = mpl.colormaps["Set1"](range(5))

    # Define the offset
    offset_v = np.arange(n_unique_sv) - n_unique_sv // 2

    fig, axs = plt.subplots(squeeze=True, layout="constrained", figsize=(8, 6))

    # Define the width of each bar based on the number of unique values
    width = (1 / (n_unique_sv)) * 0.9

    x_ticks = []
    x_tick_labels = []
    for row_idx, row in enumerate(
        df_c.group_by([first_var, second_var]).agg(pl.all()).iter_rows(named=True)
    ):
        first_coord = row[fv_c][0]
        second_coord = row[sv_c][0]

        # Move the bar by the given offset
        offset = offset_v[second_coord] * width
        for idx, (label, w, left) in enumerate(
            zip(row["variable"], row["value"], row["bottom"])
        ):
            # Hide unnecessary labels
            l = label if row_idx == 0 else "_" + label
            # Plot the bar for this specific row
            p = axs.barh(
                y=first_coord + offset,
                width=w,
                left=left,
                label=l,
                height=width,
                align="center",
                color=cmap[idx],
            )
        # Add the position and label for the ticks
        x_ticks.append(first_coord + offset)
        x_tick_labels.append(
            f"{constants.LABEL_MAPPING[first_var][row[first_var]]} {constants.LABEL_MAPPING[second_var][row[second_var]]}"
        )

        # Add the final value as label to the bar
        axs.bar_label(p, fmt="{:.2f}")
    axs.legend()
    axs.set_xlabel("Execution time (s)")
    _ = axs.set_yticks(x_ticks, x_tick_labels)

    fig.savefig(f"images/breakdown_time_{first_var}_{second_var}.png")
    # fig.savefig(f"images/breakdown_time_{first_var}_{second_var}.pdf")


def pareto_frontier_plot(
    data,
    x_var,
    y_var,
    hue_var,
    palette,
    hue_order,
    ax,
    ax_title,
    ax_xlabel,
):
    if not isinstance(data, pd.DataFrame):
        raise ValueError()
    x = data[x_var]
    y = data[y_var]

    # ax.set_xscale("log")

    xs = np.array(x)
    ys = np.array(y)
    perm = np.argsort(xs)
    xs = xs[perm]
    ys = ys[perm]
    sns.scatterplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        ax=ax,
        palette=palette,
        hue_order=hue_order,
        # legend=False
    )

    xs_pareto = [xs[0], xs[0]]
    ys_pareto = [ys[0], ys[0]]
    for i in range(1, len(xs)):
        if ys[i] > ys_pareto[-1]:
            xs_pareto.append(xs[i])
            ys_pareto.append(ys_pareto[-1])
            xs_pareto.append(xs[i])
            ys_pareto.append(ys[i])
    xs_pareto.append(ax.get_xlim()[1])
    ys_pareto.append(ys_pareto[-1])

    ax.plot(xs_pareto, ys_pareto, "--", color="k", linewidth=2, zorder=0.8)
    ax.set_ylabel("")
    # ax.set_title(ax_title)
    h, l = ax.get_legend_handles_labels()
    # ax.legend(
    #     h,
    #     [constants.LABEL_MAPPING[hue_var][_] for _ in l],
    #     title=None,
    # )
    ax.set_xlabel(ax_xlabel)

    ax.set_ylim([-0.5, 0.6])
    ax.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")

    optimal_y = ys_pareto[-1]
    return (h, l), optimal_y
