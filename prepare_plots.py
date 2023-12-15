# # %%
# %cd ~/bench
# #%%
# %load_ext autoreload
# %autoreload 2
# %%
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.gridspec import GridSpec

import src.utils.plotting as plotting
from src.utils.logging import read_logs

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)


# %%
def prepare_data_for_plotting(df: pl.DataFrame) -> pl.DataFrame:
    max_diff = df.select(pl.col("difference").abs().max()).item()
    df = df.with_columns((pl.col("difference") / max_diff).alias("scaled_diff"))
    return df


# %%
def get_cases(df: pl.DataFrame, keep_nojoin: bool = False) -> dict:
    if not keep_nojoin:
        df = df.filter(pl.col("estimator") != "nojoin")

    cases = (
        df.select(pl.col(["jd_method", "chosen_model", "estimator"]).unique().implode())
        .transpose(include_header=True)
        .to_dict(as_series=False)
    )

    return dict(zip(*list(cases.values())))


def apply_log_scaling(df, target_column, log_base=10):
    return df.with_columns(pl.col(target_column).log(log_base))


# %%
def get_difference_from_mean(
    df, column_to_average, result_column, scaled=False, geometric=False
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
    if n_unique > 2:
        prepared_df = df.join(
            df.group_by(this_groupby).agg(
                pl.mean(result_column).alias("reference_column")
            ),
            on=this_groupby,
        )

    else:
        best_method = (
            df.group_by(column_to_average)
            .agg(pl.mean(result_column))
            .top_k(1, by=result_column)[column_to_average]
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

    return prepared_df.drop(cs.ends_with("_right")).drop("reference_column")


# %%
# Prepare data for the three plot cases

# %%
root_path = Path("results/big_batch")
df_list = []
for rpath in root_path.iterdir():
    df_raw = read_logs(exp_name=None, exp_path=rpath)
    df_list.append(df_raw)

df_results = pl.concat(df_list)

# %%
df_ = df_results.select(
    pl.col(
        [
            "scenario_id",
            "target_dl",
            "jd_method",
            "base_table",
            "estimator",
            "chosen_model",
            "aggregation",
            "r2score",
            "time_fit",
            "time_predict",
            "time_run",
            "epsilon",
        ]
    )
).filter(
    (~pl.col("base_table").str.contains("open_data"))
    & (pl.col("target_dl") != "wordnet_big")
)
df_ = df_.group_by(
    ["target_dl", "jd_method", "base_table", "estimator", "chosen_model"]
).map_groups(lambda x: x.with_row_count("fold_id"))

# %%
joined = df_.join(
    df_.filter(pl.col("estimator") == "nojoin"),
    on=["target_dl", "jd_method", "base_table", "chosen_model", "fold_id"],
    how="left",
).with_columns((pl.col("r2score") - pl.col("r2score_right")).alias("difference"))

# %%
results_full = joined.filter(~pl.col("base_table").str.contains("depleted"))
results_depleted = joined.filter(pl.col("base_table").str.contains("depleted"))

results_full = prepare_data_for_plotting(results_full)
results_depleted = prepare_data_for_plotting(results_depleted)

# %%
current_results = results_depleted.clone()
projection = [
    "fold_id",
    "target_dl",
    "jd_method",
    "base_table",
    "estimator",
    "chosen_model",
    "aggregation",
    "r2score",
    "time_fit",
    "time_predict",
    "time_run",
    "difference",
    "scaled_diff",
]
current_results = current_results.select(projection)
# %%
current_results = current_results.filter(pl.col("estimator") != "nojoin")
# %%
p = get_difference_from_mean(
    current_results, column_to_average="chosen_model", result_column="r2score"
)

# %%
# Prepare comparison figure
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


# %%
df_tri = prepare_data_for_comparison(current_results, "jd_method")
# %%
plotting.draw_triple_comparison(
    df_tri,
    "estimator",
    scatterplot_dimension="chosen_model",
    figsize=(12, 4),
    scatter_mode="split",
    savefig=True,
)

# %%
cases = get_cases(
    current_results,
)

# # %%
# for outer_variable in cases:
#     plotting.draw_split_figure(cases, outer_variable, current_results)

# %%
plotting.draw_split_figure(
    cases,
    split_dimension="chosen_model",
    df=current_results,
    grouping_dimensions=["estimator"],
    kind="box",
    scatterplot_dimension="base_table",
    colormap_name="viridis",
    plotting_variable="scaled_diff",
)

# %%
target_variable = "estimator"
result_column = "scaled_diff"
_prep = get_difference_from_mean(
    current_results, column_to_average=target_variable, result_column=result_column
)

plotting.draw_split_figure(
    cases,
    split_dimension="chosen_model",
    df=_prep,
    grouping_dimensions=[target_variable],
    scatterplot_dimension="base_table",
    xtick_format="percentage",
    colormap_name="viridis",
    plotting_variable=f"diff_from_mean_{result_column}",
    # plot_label=f"Difference from mean {target_variable}",
    kind="box",
    figsize=(8, 3),
)


# %%
target_variable = "chosen_model"
result_column = "r2score"
_prep = get_difference_from_mean(
    current_results,
    column_to_average=target_variable,
    result_column=result_column,
    scaled=True,
)
cases = get_cases(
    _prep,
)

plotting.draw_split_figure(
    cases,
    split_dimension=None,
    df=_prep,
    grouping_dimensions=["jd_method"],
    scatterplot_dimension="base_table",
    scatter_mode="split",
    colormap_name="viridis",
    plotting_variable=f"diff_{target_variable}_{result_column}",
    plot_label=f"Difference between {target_variable} classes",
    kind="violin",
)

# %%
target_variable = "estimator"
result_column = "time_run"
# df = current_results.filter(pl.col("jd_method") == "exact_matching")
df = current_results
# _prep = apply_log_scaling(df, f"time_run")

_prep = get_difference_from_mean(
    df,
    column_to_average=target_variable,
    result_column=result_column,
    scaled=False,
    geometric=True,
)
# _prep = apply_log_scaling(_prep, f"diff_from_mean_time_run")

cases = get_cases(
    _prep,
)


plotting.draw_split_figure(
    cases,
    split_dimension="chosen_model",
    df=_prep,
    grouping_dimensions=[target_variable],
    scatterplot_dimension="base_table",
    plotting_variable=f"diff_from_mean_time_run",
    colormap_name="viridis",
    scatter_mode="overlapping",
    xtick_format="symlog",
    kind="box",
    figsize=(10, 5),
)

# %%
