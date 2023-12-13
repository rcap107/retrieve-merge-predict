# %%
# %cd ~/bench
# #%%
# %load_ext autoreload
# %autoreload 2
import json
import tarfile
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import polars as pl
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
        result_column,
    ]
    projection = set(projection)

    all_groupby_variables = [
        "fold_id",
        "target_dl",
        "base_table",
        "jd_method",
        "estimator",
        "chosen_model",
    ]

    this_groupby = [_ for _ in all_groupby_variables if _ != column_to_average]

    _prep = df.select(projection).join(
        df.select(projection)
        .group_by(this_groupby)
        .agg(pl.mean(result_column).alias(f"avg_{result_column}")),
        on=this_groupby,
    )

    if geometric:
        _prep = _prep.with_columns(
            (pl.col(result_column) / pl.col(f"avg_{result_column}")).alias(
                f"diff_from_mean_{result_column}"
            )
        )
    else:
        _prep = _prep.with_columns(
            (pl.col(result_column) - pl.col(f"avg_{result_column}")).alias(
                f"diff_from_mean_{result_column}"
            )
        )

    if scaled:
        _prep = _prep.with_columns(
            _prep.with_columns(
                pl.col(f"diff_from_mean_{result_column}")
                / pl.col(f"diff_from_mean_{result_column}").abs().max()
            )
        )

    return _prep


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
cases = get_cases(
    current_results,
)

# # %%
# for outer_variable in cases:
#     plotting.draw_plot(cases, outer_variable, current_results)

# %%
plotting.draw_plot(
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

plotting.draw_plot(
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
target_variable = "estimator"
result_column = "r2score"
_prep = get_difference_from_mean(
    current_results.filter(pl.col("chosen_model") == "catboost"),
    column_to_average=target_variable,
    result_column=result_column,
    scaled=True,
)
cases = get_cases(
    _prep,
)

plotting.draw_plot(
    cases,
    split_dimension="chosen_model",
    df=_prep,
    grouping_dimensions=[target_variable],
    scatterplot_dimension="base_table",
    colormap_name="viridis",
    plotting_variable=f"diff_from_mean_{result_column}",
    # plot_label=f"Difference from mean {target_variable}",
    kind="box",
)

# %%
# %%
target_variable = "estimator"
result_column = "time_run"
# df = current_results.filter(pl.col("chosen_model") == "catboost")
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


plotting.draw_plot(
    cases,
    split_dimension="chosen_model",
    df=_prep,
    grouping_dimensions=[target_variable],
    scatterplot_dimension="base_table",
    plotting_variable=f"diff_from_mean_time_run",
    colormap_name="viridis",
    xtick_format="linear",
    kind="box",
)

# %%
# %%
target_variable = "estimator"
result_column = "time_run"
# df = current_results.filter(pl.col("chosen_model") == "catboost")
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


plotting.draw_plot(
    cases,
    split_dimension="estimator",
    df=_prep,
    grouping_dimensions=[target_variable],
    scatterplot_dimension="base_table",
    plotting_variable=f"diff_from_mean_time_run",
    colormap_name="viridis",
    xtick_format="linear",
    kind="box",
)

# %%
