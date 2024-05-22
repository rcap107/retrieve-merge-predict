# %%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2
# %%
import polars as pl

from src.utils import plotting
from src.utils.logging import read_and_process

# %%
result_path = "results/overall/overall_first.parquet"

# Use the standard method for reading all results for consistency.
df_results = pl.read_parquet(result_path)
current_results = read_and_process(df_results)
current_results = current_results.filter(pl.col("estimator") != "nojoin")
# %%
# We remove top_k_full_join and starmie from this set of results.
_d = current_results.filter(
    (pl.col("estimator") != "top_k_full_join") & (pl.col("jd_method") != "starmie")
)
plot_case = "dep"

# Set to false to plot the results without saving the figure on disk.
savefig = True
# %% Selector plot
var = "estimator"
scatter_d = "case"
plotting.draw_pair_comparison(
    _d,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 2.1),
    scatter_mode="split",
    savefig=savefig,
    savefig_type=["png", "pdf"],
    case=plot_case,
    jitter_factor=0.02,
    qle=0.05,
    add_titles=True,
    sorting_method="manual",
    sorting_variable="estimator_comp",
)
# %% ML Model
var = "chosen_model"
scatter_d = "case"
plotting.draw_pair_comparison(
    _d,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 1),
    scatter_mode="split",
    savefig=savefig,
    savefig_type=["png", "pdf"],
    case=plot_case,
    add_titles=False,
)
