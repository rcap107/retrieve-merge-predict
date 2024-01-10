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
root_path = Path("results/big_batch")
df_list = []
for rpath in root_path.iterdir():
    df_raw = read_logs(exp_name=None, exp_path=rpath)
    df_list.append(df_raw)

df_results = pl.concat(df_list)

#%%
results_full, results_depleted = read_and_process(df_results)


# %%
case = "dep"

if case == "dep":
    current_results = results_depleted.clone()
    current_results = current_results.filter(pl.col("estimator") != "nojoin")

elif case == "full":
    current_results = results_full.clone()

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
axes[3].spines["top"].set_visible(False)
axes[3].spines["right"].set_visible(False)
axes[3].set_xticks([])
axes[3].set_yticks([])

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
