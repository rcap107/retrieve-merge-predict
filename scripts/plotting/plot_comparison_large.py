"""
Alternative code for Figure 5: comparing across data lakes over selector, aggregation and ML model.
"""

# %%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt

# %%
import polars as pl

from src.utils import constants, plotting

# %%
plot_case = "dep"

savefig = False
# %%
_results_general = (
    pl.read_parquet("results/temp_results_general.parquet")
    .with_columns(
        case=(
            pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
        )
    )
    .with_columns(
        prediction_metric=pl.when(pl.col("prediction_metric") < -1)
        .then(-1)
        .otherwise(pl.col("prediction_metric"))
    )
)
_results_aggr = (
    pl.read_parquet("results/temp_results_aggregation.parquet")
    .with_columns(
        case=(
            pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
        )
    )
    .with_columns(
        prediction_metric=pl.when(pl.col("prediction_metric") < -1)
        .then(-1)
        .otherwise(pl.col("prediction_metric"))
    )
)

_results_retrieval = (
    pl.read_parquet("results/temp_results_retrieval.parquet")
    .filter(pl.col("estimator") != "nojoin")
    .with_columns(
        case=(
            pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
        )
    )
    .with_columns(
        prediction_metric=pl.when(pl.col("prediction_metric") < -1)
        .then(-1)
        .otherwise(pl.col("prediction_metric"))
    )
)


# %%
fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 6.5),
    layout="constrained",
    squeeze=True,
    height_ratios=(2, 2, 2, 2),
)

var = "jd_method"
scatter_d = "case"

subplot_titles = ["a. Retrieval method", ""]
plotting.draw_pair_comparison(
    _results_retrieval,
    var,
    scatterplot_dimension=scatter_d,
    scatter_mode="split",
    savefig=savefig,
    savefig_type=["png", "pdf"],
    case=plot_case,
    jitter_factor=0.02,
    qle=0.05,
    add_titles=True,
    # sorting_method="manual",
    # sorting_variable="estimator_comp",
    axes=axes[0, :],
    subplot_titles=subplot_titles,
    figure=fig,
)


var = "estimator"
scatter_d = "case"

subplot_titles = ["b. Join selection method", ""]
plotting.draw_pair_comparison(
    _results_general,
    var,
    scatterplot_dimension=scatter_d,
    scatter_mode="split",
    savefig=savefig,
    savefig_type=["png", "pdf"],
    case=plot_case,
    jitter_factor=0.02,
    qle=0.05,
    add_titles=True,
    sorting_method="manual",
    sorting_variable="estimator_comp",
    axes=axes[1, :],
    subplot_titles=subplot_titles,
    figure=fig,
)

var = "aggregation"
subplot_titles = ["c. Aggregation method", ""]

plotting.draw_pair_comparison(
    _results_aggr,
    var,
    scatterplot_dimension=scatter_d,
    scatter_mode="split",
    savefig=savefig,
    savefig_type=["png", "pdf"],
    case=plot_case,
    colormap_name="Set1",
    jitter_factor=0.03,
    qle=0.01,
    add_titles=True,
    axes=axes[2, :],
    subplot_titles=subplot_titles,
    figure=fig,
)


var = "chosen_model"
scatter_d = "case"
subplot_titles = ["d. Supervised learner", ""]
plotting.draw_pair_comparison(
    _results_general,
    var,
    scatterplot_dimension=scatter_d,
    scatter_mode="split",
    savefig=savefig,
    savefig_type=["png", "pdf"],
    case=plot_case,
    jitter_factor=0.02,
    qle=0.05,
    add_titles=True,
    axes=axes[3, :],
    subplot_titles=subplot_titles,
    figure=fig,
)


for _ in range(4):
    # axes[_, 1].sharey(axes[_, 0])
    axes[_, 1].set_yticks([])

# %%
# fig.savefig("images/dep_pair_full.png")
# fig.savefig("images/dep_pair_full.pdf")

# %%
