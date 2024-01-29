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
from src.utils.logging import read_and_process, read_logs

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

# %%
result_path = "results/overall/wordnet_aggr.parquet"

df_results = pl.read_parquet(result_path)

results_full, results_depleted = read_and_process(df_results)


# %%
case = "dep"

if case == "dep":
    current_results = results_depleted.clone()
    current_results = current_results.filter(pl.col("estimator") != "nojoin")

elif case == "full":
    current_results = results_full.clone()

# %%
var = "estimator"
scatter_d = "base_table"
plotting.draw_pair_comparison(
    current_results,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 3),
    scatter_mode="split",
    savefig=False,
    savefig_type=["png", "pdf"],
    case=case,
    colormap_name="Set1",
)

# %%
