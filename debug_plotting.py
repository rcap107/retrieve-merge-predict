# %%
# %cd ~/bench
# %load_ext autoreload
# %autoreload 2
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import src.utils.plotting as plotting
from src.utils.logging import read_and_process

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

# %%
result_path = "results/overall/overall_first.parquet"

df_results = pl.read_parquet(result_path)

current_results = read_and_process(df_results)
current_results = current_results.filter(pl.col("estimator") != "nojoin")
# %%
_d = current_results.filter(
    (pl.col("estimator") != "top_k_full_join")
    & (pl.col("jd_method") != "starmie")
    # & (~pl.col("target_dl").is_in(["wordnet_vldb_50","open_data_us"]) )
)
# %%
plot_case = "dep"
var = "estimator"
scatter_d = "case"
plotting.draw_pair_comparison(
    _d,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 2.1),
    scatter_mode="split",
    savefig=True,
    savefig_type=["png", "pdf"],
    case=plot_case,
    jitter_factor=0.01,
    qle=0.05,
    add_titles=True,
    # colormap_name="Set1",
)
# %%
var = "chosen_model"
scatter_d = "case"
plotting.draw_pair_comparison(
    _d,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 1),
    scatter_mode="split",
    savefig=True,
    savefig_type=["png", "pdf"],
    case=plot_case,
    add_titles=False,
    # colormap_name="Set1",
)

# %%
