"""
This script is used to prepare the special case of aggregation in figure 5(c) in the paper.
"""

# %%
# %cd ~/bench

# %%
import polars as pl

from src.utils import constants, plotting
from src.utils.logging import read_and_process

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

# %%
this_case = "dep"
target_tables = [
    "company_employees",
    # "housing_prices",
    # "movies_large",
    "us_accidents_2021",
    # "us_accidents_large",
    "us_county_population",
    # "us_elections",
    "schools",
]


# %%
aggr_result_path = "stats/overall/overall_aggr.parquet"
df_results = pl.read_parquet(aggr_result_path)
current_results = read_and_process(df_results)
others = [
    col
    for col in current_results.columns
    if col not in constants.GROUPING_KEYS + ["case"]
]
results_aggr = current_results.group_by(constants.GROUPING_KEYS + ["case"]).agg(
    pl.mean(others)
)

results_aggr = results_aggr.filter(
    (pl.col("estimator").is_in(["best_single_join", "highest_containment"]))
    & (pl.col("jd_method") != "starmie")
)

results_aggr = results_aggr.filter(
    (pl.col("base_table").str.split("-").list.first()).is_in(target_tables)
)

# %%
var = "aggregation"
scatter_d = "case"
plotting.draw_pair_comparison(
    results_aggr,
    var,
    scatterplot_dimension=scatter_d,
    figsize=(10, 2.1),
    scatter_mode="split",
    savefig=False,
    savefig_type=["png", "pdf"],
    case=this_case,
    colormap_name="Set1",
    jitter_factor=0.03,
    qle=0.01,
    add_titles=False,
)
# %%
