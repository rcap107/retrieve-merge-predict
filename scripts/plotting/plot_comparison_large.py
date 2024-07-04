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

from src.utils import plotting
from src.utils.logging import read_and_process


def prepare_general():
    result_path = "stats/overall/overall_first.parquet"

    # Use the standard method for reading all results for consistency.
    df_results = pl.read_parquet(result_path)
    current_results = read_and_process(df_results)
    current_results = current_results.filter(pl.col("estimator") != "nojoin")
    _d = current_results.filter(
        (pl.col("estimator") != "top_k_full_join") & (pl.col("jd_method") != "starmie")
    )
    return _d


def prepare_aggr():
    target_tables = [
        "company_employees",
        "movies_large",
        "us_accidents_2021",
        "us_county_population",
        "schools",
    ]
    aggr_result_path = "results/overall/overall_aggr.parquet"
    df_results = pl.read_parquet(aggr_result_path)
    results_aggr = read_and_process(df_results)
    results_aggr = results_aggr.filter(
        (pl.col("estimator").is_in(["best_single_join", "highest_containment"]))
        & (pl.col("jd_method") != "starmie")
    )

    results_aggr = results_aggr.filter(
        (pl.col("base_table").str.split("-").list.first()).is_in(target_tables)
    )
    return results_aggr


# %%
plot_case = "dep"

savefig = False
# %%
_results_general = prepare_general()
_results_aggr = prepare_aggr()
# %%
fig, axes = plt.subplots(
    3, 2, figsize=(10, 6.5), layout="constrained", squeeze=True, height_ratios=(2, 2, 1)
)

var = "estimator"
scatter_d = "case"

subplot_titles = ["a. Join selection method", ""]
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
    axes=axes[0, :],
    subplot_titles=subplot_titles,
)

var = "chosen_model"
scatter_d = "case"
subplot_titles = ["c. Supervised learner", ""]
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
    axes=axes[2, :],
    subplot_titles=subplot_titles,
)

var = "aggregation"
subplot_titles = ["b. Aggregation method", ""]

plotting.draw_pair_comparison(
    _results_aggr,
    var,
    scatterplot_dimension=scatter_d,
    scatter_mode="split",
    savefig=False,
    savefig_type=["png", "pdf"],
    case=plot_case,
    colormap_name="Set1",
    jitter_factor=0.03,
    qle=0.01,
    add_titles=True,
    axes=axes[1, :],
    subplot_titles=subplot_titles,
)

for _ in range(3):
    # axes[_, 1].sharey(axes[_, 0])
    axes[_, 1].set_yticks([])

# %%
