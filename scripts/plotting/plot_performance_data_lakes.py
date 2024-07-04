"""
Figure 7: prediction performance by data lake.
"""
# %%
# %cd ..
# %load_ext autoreload
# %autoreload 2
#%%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.utils import constants, plotting
from src.utils.logging import read_and_process

# %%
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")
plt.rc("font", family="sans-serif")

# %%
result_path = "stats/overall/overall_first.parquet"

df_results = pl.read_parquet(result_path)

current_results = read_and_process(df_results)
df_raw = current_results.filter(pl.col("estimator") != "nojoin")

# %%

fig, axs = plt.subplots(
    1, 1, squeeze=True, figsize=(5, 2), layout="constrained", sharex=False
)

var_to_plot = "y"

plotting.prepare_case_subplot(
    axs,
    df=df_raw,
    grouping_dimension="target_dl",
    scatterplot_dimension=None,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_method="manual",
    sorting_variable="target_dl",
    jitter_factor=0.05,
    scatter_mode="split",
    qle=0,
    xtick_format="linear",
)
fig.savefig("images/prediction_performance.png")
fig.savefig("images/prediction_performance.pdf")


# %%
