from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns

plt.style.use("seaborn-v0_8-talk")

GROUPING_KEYS = ["jd_method", "estimator", "chosen_model", "target_dl", "base_table"]

LABEL_MAPPING = {
    "base_table": {
        "company-employees-yadl-depleted": "(D) Employees",
        "company_employees-yadl-depleted": "(D) Employees",
        "movies-yadl-depleted": "(D) Movies",
        "movies-vote-yadl-depleted": "(D) Movies Vote",
        "movies_vote-yadl-depleted": "(D) Movies Vote",
        "housing-prices-yadl-depleted": "(D) Housing Prices",
        "housing_prices-yadl-depleted": "(D) Housing Prices",
        "us-accidents-yadl-depleted": "(D) US Accidents",
        "us_accidents-yadl-depleted": "(D) US Accidents",
        "us-elections-yadl-depleted": "(D) US Elections",
        "us_elections-yadl-depleted": "(D) US Elections",
        "us_county_population-depleted-yadl": "(D) US County Population",
        "us_county_population-yadl-depleted": "(D) US County Population",
        "company-employees-yadl": "Employees",
        "company_employees-yadl": "Employees",
        "movies-yadl": "Movies",
        "movies-vote-yadl": "Movies Vote",
        "movies_vote-yadl": "Movies Vote",
        "housing-prices-yadl": "Housing Prices",
        "housing_prices-yadl": "Housing Prices",
        "us-accidents-yadl": "US Accidents",
        "us_accidents-yadl": "US Accidents",
        "us-elections-yadl": "US Elections",
        "us_elections-yadl": "US Elections",
    },
    "jd_method": {
        "exact_matching": "Exact",
        "minhash": "MinHash",
        "minhash_hybrid": "Hybrid MinHash",
    },
    "chosen_model": {"catboost": "CatBoost", "linear": "Linear"},
    "estimator": {
        "full_join": "Full Join",
        "best_single_join": "Best Single Join",
        "stepwise_greedy_join": "Stepwise Greedy Join",
        "highest_containment": "Highest Cont. Join",
        "nojoin": "No Join",
    },
    "variables": {
        "estimator": "Estimator",
        "jd_method": "Retrieval method",
        "chosen_model": "ML model",
        "base_table": "Base table",
    },
    "aggregation": {"first": "First", "mean": "Mean", "DFS": "DFS"},
    "budget_amount": {10: 10, 30: 30, 100: 100},
}


def get_difference_from_mean(
    df,
    column_to_average,
    result_column,
    scaled=False,
    geometric=False,
    force_split=False,
):
    all_groupby_variables = [
        "fold_id",
        "target_dl",
        "base_table",
        "jd_method",
        "estimator",
        "chosen_model",
    ]

    this_groupby = [_ for _ in all_groupby_variables if _ != column_to_average]

    n_unique = df.select(pl.col(column_to_average).n_unique()).item()
    if n_unique > 2 or force_split:
        prepared_df = df.join(
            df.group_by(this_groupby).agg(
                pl.mean(result_column).alias("reference_column")
            ),
            on=this_groupby,
        )

    else:
        best_method = (
            df.group_by(column_to_average)
            .agg(pl.mean("scaled_diff"))
            .top_k(1, by="scaled_diff")[column_to_average]
            .item()
        )

        prepared_df = (
            df.filter(pl.col(column_to_average) == best_method)
            .join(df, on=this_groupby)
            .filter(pl.col(column_to_average) != pl.col(column_to_average + "_right"))
            .rename({result_column + "_right": "reference_column"})
        )

    if geometric:
        prepared_df = prepared_df.with_columns(
            (pl.col(result_column) / pl.col("reference_column")).alias(
                f"diff_{column_to_average}_{result_column}"
            )
        )
    else:
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


def prepare_data_for_comparison(df, variable_of_interest):
    df = get_difference_from_mean(
        df, column_to_average=variable_of_interest, result_column="r2score"
    )
    df = get_difference_from_mean(
        df,
        column_to_average=variable_of_interest,
        result_column="time_run",
        geometric=True,
    )

    return df


def prepare_jitter(shape, offset_value, factor=0.1):
    data = np.ones(shape) * offset_value
    return data + np.random.normal(0, factor, data.shape)


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


def prepare_scatterplot_labels(
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
    # colors = plt.colormaps[colormap_name](np.linspace(0, 1, len(scatterplot_labels)))
    colors = plt.colormaps[colormap_name].resampled(len(scatterplot_labels)).colors
    scatterplot_mapping = dict(
        zip(
            scatterplot_labels,
            colors,
        )
    )
    return scatterplot_mapping


def _custom_formatter(x, pos):
    return rf"{x:g}x"


def format_xaxis(ax, case, xmax=1):
    if case == "percentage":
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=xmax, decimals=0))
        # ax.xaxis.set_minor_formatter(ticker.PercentFormatter(xmax=xmax, decimals=0))
    elif case == "log":
        ax.set_xscale("log", base=10)
        ax.xaxis.set_major_locator(
            ticker.LogLocator(
                base=2,
            )
        )
        minor_locator = ticker.LogLocator(base=2)
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
            linthresh=2,
            subs=[
                0.5,
                1,
            ],
        )
        # major_locator = ticker.FixedLocator([0, 0.5, 1, 1.5, 2, 3])
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_custom_formatter))
    return ax


def base_barplot(
    df: pd.DataFrame,
    categorical_variable: str = "estimator",
    result_variable: str = "r2score",
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
        y_variable (str, optional): Numerical variable to plot. Defaults to "r2score".
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


def prepare_case_subplot(
    ax,
    df,
    grouping_dimension,
    scatterplot_dimension,
    plotting_variable,
    kind="box",
    xtick_format="percentage",
    scatter_mode="overlapping",
    jitter_factor=0.05,
    scatterplot_mapping=None,
    colormap_name="viridis",
    scatterplot_marker_size=3,
    average_folds="none",
    box_width=0.9,
    xmax=1,
    sorting_variable="r2score",
):
    # Prepare the plotting data sorting by `sorting_variable` (r2score by default to have  consistency over axes)
    if sorting_variable == plotting_variable:
        data = (
            df.select(grouping_dimension, sorting_variable)
            .group_by(grouping_dimension)
            .agg(pl.all(), pl.mean(sorting_variable).alias("sort_col"))
            .sort("sort_col")
            .to_dict()
        )
    else:
        data = (
            df.select(grouping_dimension, plotting_variable, sorting_variable)
            .group_by(grouping_dimension)
            .agg(pl.all(), pl.mean(sorting_variable).alias("sort_col"))
            .sort("sort_col")
            .select(grouping_dimension, plotting_variable)
            .to_dict()
        )

    if scatterplot_mapping is None:
        scatterplot_mapping = prepare_scatterplot_labels(
            df, scatterplot_dimension, plotting_variable, colormap_name=colormap_name
        )

    ref_vline = 1 if xtick_format in ["log", "symlog"] else 0

    ax.axvline(ref_vline, alpha=0.4, zorder=0, color="blue", linestyle="--")
    if kind == "violin":
        parts = ax.violinplot(
            data[plotting_variable],
            showmedians=False,
            showmeans=False,
            vert=False,
        )
        for pc in parts["bodies"]:
            pc.set_edgecolor("black")
            pc.set_facecolor("none")
            pc.set_alpha(1)
            pc.set_linewidth(2)
            pc.set_zorder(2)
    elif kind == "box":
        medianprops = dict(linewidth=2, color="red")
        whiskerprops = dict(linewidth=2)
        capprops = dict(linewidth=2)
        boxprops = dict(facecolor="white", linewidth=2)
        bp = ax.boxplot(
            data[plotting_variable],
            showfliers=False,
            vert=False,
            widths=box_width,
            medianprops=medianprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            zorder=2,
            patch_artist=True,
        )

    facecolors = ["grey", "white"]
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
        if scatter_mode == "split":
            offset = (
                np.linspace(
                    -box_width / 2 + jitter_factor,
                    box_width / 2 - jitter_factor,
                    len(scatterplot_mapping),
                )
            ) * 0.8
        elif scatter_mode == "overlapping":
            offset = np.zeros(len(scatterplot_mapping))

        for _, label in enumerate(scatterplot_mapping):
            this_label = LABEL_MAPPING[scatterplot_dimension][label]
            if _i > 0:
                this_label = "_" + str(this_label)
            filter_dict = {
                scatterplot_dimension: label,
                grouping_dimension: data[grouping_dimension][_i],
            }
            values = df.filter(**filter_dict)[plotting_variable].to_numpy()

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
                alpha=0.7,
                label=this_label,
                zorder=2.5,
            )
    h, l = ax.get_legend_handles_labels()
    if len(data[plotting_variable]) == 1:
        ax.set_yticks([1], ["Best"])
    else:
        ax.set_yticks(
            range(1, len(data[grouping_dimension]) + 1),
            [LABEL_MAPPING[grouping_dimension][_l] for _l in data[grouping_dimension]],
        )

    ax = format_xaxis(ax, xtick_format, xmax)

    xlim = ax.get_xlim()
    if xtick_format in ["symlog", "log"]:
        ax.axvspan(1, xlim[1], zorder=0, alpha=0.05, color="red")
    else:
        ax.axvspan(xlim[0], 0, zorder=0, alpha=0.05, color="red")
    ax.set_xlim(xlim)

    return h, l


def draw_split_figure(
    cases: dict,
    df: pl.DataFrame,
    split_dimension: str | None = None,
    grouping_dimensions: str | list[str] | None = None,
    scatterplot_dimension: str = "estimator",
    plotting_variable: str = "scaled_diff",
    kind: str = "violin",
    axes_formatting: dict = None,
    xtick_format: str = "percentage",
    colormap_name: str = "viridis",
    scatter_mode: str = "overlapping",
    plot_label: str = None,
    figsize=(8, 3),
):
    if axes_formatting is None:
        axes_formatting = {
            "xaxis": {
                "xtick_format": "percentage",
                "xmax": 1,
                "logscale_base": 2,
            },
            "yaxis": {},
        }

    # Inner variables are all the variables that will be plotted, except the outer variable and the scatterplot variable
    if grouping_dimensions is None:
        grouping_dimensions = [
            _
            for _ in cases.keys()
            if (_ != split_dimension) & (_ != scatterplot_dimension)
        ]
    if isinstance(grouping_dimensions, str):
        grouping_dimensions = [grouping_dimensions]

    # Check that all columns are found
    assert all(_ in df.columns for _ in grouping_dimensions)
    if split_dimension is not None:
        assert split_dimension in df.columns
    assert scatterplot_dimension in df.columns
    assert plotting_variable in df.columns
    assert plotting_variable is not None

    if split_dimension is not None:
        n_cols = len(cases[split_dimension])
    else:
        n_cols = 1
    scatterplot_mapping = prepare_scatterplot_labels(
        df, scatterplot_dimension, plotting_variable, colormap_name
    )

    for idx_inner_var, case_grouping_ in enumerate(grouping_dimensions):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_cols,
            figsize=figsize,
            sharex=True,
            sharey="row",
            layout="constrained",
            squeeze=False,
        )

        if split_dimension is not None:
            for idx_outer_var, case_split_ in enumerate(cases[split_dimension]):
                ax = axes[0, idx_outer_var]
                subset = df.filter(pl.col(split_dimension) == case_split_)

                h, l = prepare_case_subplot(
                    ax,
                    subset,
                    case_grouping_,
                    scatterplot_dimension,
                    plotting_variable,
                    scatterplot_mapping=scatterplot_mapping,
                    scatter_mode=scatter_mode,
                    xtick_format=xtick_format,
                    kind=kind,
                )
                axes[0][idx_outer_var].set_title(
                    LABEL_MAPPING[split_dimension][case_split_]
                )
        else:
            ax = axes[0, 0]
            h, l = prepare_case_subplot(
                ax,
                df,
                case_grouping_,
                scatterplot_dimension,
                plotting_variable,
                scatterplot_mapping=scatterplot_mapping,
                scatter_mode=scatter_mode,
                xtick_format=xtick_format,
                kind=kind,
            )
            # axes[0][0].set_title(
            #     LABEL_MAPPING[split_dimension][case_outer_]
            # )

        fig.legend(
            h,
            l,
            # loc="outside right",
            loc="outside lower left",
            mode="expand",
            ncols=len(l),
            markerscale=5,
            borderaxespad=-0.2,
            bbox_to_anchor=(0, -0.1, 1, 0.5),
            scatterpoints=1,
        )
        fig.set_constrained_layout_pads(
            w_pad=5.0 / 72.0, h_pad=4.0 / 72.0, hspace=0.0 / 72.0, wspace=0.0 / 72.0
        )
        # fig.suptitle(outer_dimension)
        if plot_label is not None:
            fig.supxlabel(plot_label)

        fig.savefig("test.pdf")


def draw_triple_comparison(
    df,
    grouping_dimension,
    scatterplot_dimension,
    form_factor: str = "multi",
    scatter_mode="overlapping",
    colormap_name="viridis",
    figsize=(10, 4),
    savefig: bool = False,
    savefig_type: list | str = "png",
    case: str = "dep",
):
    # The form factor decides whether plots should be on a row, or use a gridspec
    assert form_factor in ["multi", "binary"]

    df_rel_r2 = get_difference_from_mean(
        df, column_to_average=grouping_dimension, result_column="r2score"
    )
    df_time = get_difference_from_mean(
        df,
        column_to_average=grouping_dimension,
        result_column="time_run",
        geometric=True,
    )

    scatterplot_mapping = prepare_scatterplot_labels(
        df, scatterplot_dimension, "scaled_diff", colormap_name
    )

    if form_factor == "multi":
        fig, axes = plt.subplot_mosaic(
            [[1, 2, 3]], layout="constrained", sharey=True, figsize=figsize
        )
    else:
        fig, axes = plt.subplot_mosaic(
            [[1, 2], [1, 3]], layout="constrained", sharey=False, figsize=figsize
        )

    plotting_variables = [
        "scaled_diff",
        f"diff_{grouping_dimension}_r2score",
        f"diff_{grouping_dimension}_time_run",
    ]

    formatting_dict = {
        "scaled_diff": {"xtick_format": "percentage"},
        f"diff_{grouping_dimension}_r2score": {"xtick_format": "percentage"},
        f"diff_{grouping_dimension}_time_run": {"xtick_format": "symlog"},
    }

    subplot_titles = [
        rf"% $R^2$ difference",
        rf"Relative $R^2$ ",
        rf"Relative computation time",
    ]

    if scatter_mode is None:
        scatter_mode = "split" if len(scatterplot_mapping) > 2 else "overlapping"

    plot_df = [df, df_rel_r2, df_time]
    for idx, var in enumerate(plotting_variables, start=0):
        ax = axes[idx + 1]
        # ax.grid(which="both", axis="x", alpha=0.3)
        h, l = prepare_case_subplot(
            ax,
            plot_df[idx],
            grouping_dimension,
            scatterplot_dimension,
            var,
            scatterplot_mapping=scatterplot_mapping,
            scatter_mode=scatter_mode,
            xtick_format=formatting_dict[var]["xtick_format"],
            kind="box",
            jitter_factor=0.03,
        )
        axes[idx + 1].set_title(subplot_titles[idx])

    # fig.legend(
    #     h,
    #     l,
    #     loc="right",
    #     ncols=1,
    #     markerscale=5,
    #     # borderaxespad=-0.2,
    #     # bbox_to_anchor=(0, -0.1, 1, 0.5),
    #     scatterpoints=1,
    # )
    # fig.legend(
    #     h,
    #     l,
    #     # loc="outside right",
    #     loc="lower left",
    #     mode="expand",
    #     ncols=len(l),
    #     markerscale=5,
    #     # borderaxespad=-0.2,
    #     # bbox_to_anchor=(0, -0.1, 1, 0.5),
    #     scatterpoints=1,
    # )
    fig.set_constrained_layout_pads(
        w_pad=5.0 / 72.0, h_pad=4.0 / 72.0, hspace=0.0 / 72.0, wspace=0.0 / 72.0
    )
    # fig.suptitle(
    #     f"{LABEL_MAPPING['variables'][grouping_dimension]} - {LABEL_MAPPING['variables'][scatterplot_dimension]}"
    # )

    if savefig:
        if isinstance(savefig_type, str):
            savefig_type = [savefig_type]
        for ext in savefig_type:
            fname = f"{case}_triple_{grouping_dimension}_{scatterplot_dimension}.{ext}"
            print(fname)
            fig.savefig(Path("images", fname))


def draw_pair_comparison(
    df,
    grouping_dimension,
    scatterplot_dimension,
    form_factor="multi",
    scatter_mode="overlapping",
    colormap_name="viridis",
    figsize=(10, 4),
    savefig: bool = False,
    savefig_type: list | str = "png",
    case: str = "dep",
):
    df_rel_r2 = get_difference_from_mean(
        df, column_to_average=grouping_dimension, result_column="r2score"
    )
    df_time = get_difference_from_mean(
        df,
        column_to_average=grouping_dimension,
        result_column="time_run",
        geometric=True,
    )

    scatterplot_mapping = prepare_scatterplot_labels(
        df, scatterplot_dimension, "scaled_diff", colormap_name
    )

    if form_factor == "multi":
        fig, axes = plt.subplot_mosaic(
            [[0, 1]],
            layout="constrained",
            figsize=figsize,
            sharey=True,
            width_ratios=(2, 2),
        )
    elif form_factor == "binary":
        fig, axes = plt.subplot_mosaic(
            [
                [
                    0,
                ],
                [
                    1,
                ],
            ],
            layout="constrained",
            figsize=figsize,
            sharey=False,
            width_ratios=(1),
        )

    axes[1].sharey(axes[0])
    axes[1].set_yticks([])

    plotting_variables = [
        f"diff_{grouping_dimension}_r2score",
        f"diff_{grouping_dimension}_time_run",
    ]

    formatting_dict = {
        f"diff_{grouping_dimension}_r2score": {"xtick_format": "percentage"},
        f"diff_{grouping_dimension}_time_run": {"xtick_format": "symlog"},
    }

    subplot_titles = [
        rf"$R^2$ difference",
        rf"Time difference",
    ]

    if scatter_mode is None:
        scatter_mode = "split" if len(scatterplot_mapping) > 2 else "overlapping"

    plot_df = [df_rel_r2, df_time]
    for idx, var in enumerate(plotting_variables, start=0):
        ax = axes[idx]
        # ax.grid(which="both", axis="x", alpha=0.3)
        h, l = prepare_case_subplot(
            ax,
            plot_df[idx],
            grouping_dimension,
            scatterplot_dimension,
            var,
            scatterplot_mapping=scatterplot_mapping,
            scatter_mode=scatter_mode,
            xtick_format=formatting_dict[var]["xtick_format"],
            kind="box",
            jitter_factor=0.03,
        )
        axes[idx].set_title(subplot_titles[idx])

    fig.legend(
        h,
        l,
        loc="outside right",
        ncols=1,
        markerscale=2,
        # borderaxespad=-0.2,
        # bbox_to_anchor=(0, -0.1, 1, 0.5),
        scatterpoints=1,
    )
    # fig.legend(
    #     h,
    #     l,
    #     # loc="outside right",
    #     loc="lower left",
    #     mode="expand",
    #     ncols=len(l),
    #     markerscale=5,
    #     # borderaxespad=-0.2,
    #     # bbox_to_anchor=(0, -0.1, 1, 0.5),
    #     scatterpoints=1,
    # )
    fig.set_constrained_layout_pads(
        w_pad=5.0 / 72.0, h_pad=4.0 / 72.0, hspace=0.0 / 72.0, wspace=5.0 / 72.0
    )
    # fig.suptitle(
    #     f"{LABEL_MAPPING['variables'][grouping_dimension]} - {LABEL_MAPPING['variables'][scatterplot_dimension]}"
    # )

    if savefig:
        if isinstance(savefig_type, str):
            savefig_type = [savefig_type]
        for ext in savefig_type:
            fname = f"{case}_triple_{grouping_dimension}_{scatterplot_dimension}.{ext}"
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

    # Find the number of unique values in the second variable
    n_unique = df_c[sv_c].n_unique()

    # Prepare a palette with 5 colors (one for each step of the pipeline)
    cmap = mpl.colormaps["Set1"](range(5))

    # Define the offset
    offset_v = np.arange(n_unique) - n_unique // 2

    fig, axs = plt.subplots(squeeze=True, layout="constrained")

    # Define the width of each bar based on the number of unique values
    width = 1 / (n_unique + 1)

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
            f"{LABEL_MAPPING[first_var][row[first_var]]} {LABEL_MAPPING[second_var][row[second_var]]}"
        )

        # Add the final value as label to the bar
        axs.bar_label(p, fmt="{:.2f}")
    axs.legend()
    axs.set_xlabel("Execution time (s)")
    _ = axs.set_yticks(x_ticks, x_tick_labels)

    fig.savefig(f"images/breakdown_time_{first_var}_{second_var}.pdf")
