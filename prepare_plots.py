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

#%%

# Create a figure and a GridSpec with 2 rows and 2 columns
fig = plt.figure(figsize=(8, 3))
gs = GridSpec(2, 2, width_ratios=[2, 1])

# Add subplots to the GridSpec
ax_left = fig.add_subplot(gs[:, 0])  # One plot in the left column, spanning both rows
ax_top_right = fig.add_subplot(gs[0, 1])  # Top plot in the right column
ax_bottom_right = fig.add_subplot(
    gs[1, 1], sharey=ax_top_right
)  # Bottom plot in the right column

# Customize the subplots (you can plot your data here)
ax_left.set_title("Left Plot")
ax_top_right.set_title("Top Right Plot")
ax_bottom_right.set_title("Bottom Right Plot")

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot
plt.show()

# %%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig, axes = plt.subplot_mosaic(
    [[1, 3], [2, 3]],
    layout="constrained",
    figsize=(8, 3),
    sharey=True,
    width_ratios=(3, 1),
)
axes[3].set_frame_on(False)


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


axes[3] = simpleaxis(axes[3])

# %%
