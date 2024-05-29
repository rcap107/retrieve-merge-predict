"""Figure 6: Relative prediction performance using top-1 full join.
"""

# #%%
# %cd ~/bench
# #%%
# %load_ext autoreload
# %autoreload 2
# %%
import matplotlib.pyplot as plt
import polars as pl

import src.utils.plotting as plotting
from src.utils.logging import read_and_process

# %%
result_path = "results/overall/overall_first.parquet"
df_results = pl.read_parquet(result_path)

results_depleted = read_and_process(df_results)
_d = results_depleted.filter(
    (pl.col("estimator") == "top_k_full_join")
    # Selecting only the data lakes that could be run with Starmie
    & (~pl.col("target_dl").is_in(["wordnet_vldb_50", "open_data_us"]))
)
scatter_d = "case"

df_rel_r2 = plotting.get_difference_from_mean(
    _d, column_to_average="jd_method", result_column="y"
)
scatterplot_mapping = plotting.prepare_scatterplot_mapping_case(_d)

# %%
fig, ax = plt.subplots(squeeze=True, layout="constrained", figsize=(4.5, 2))

var_to_plot = "diff_jd_method_y"

plotting.prepare_case_subplot(
    ax,
    df=df_rel_r2,
    grouping_dimension="jd_method",
    scatterplot_dimension=None,
    plotting_variable=var_to_plot,
    kind="box",
    sorting_method="manual",
    sorting_variable="jd_method",
    jitter_factor=0.05,
    scatter_mode="split",
    scatterplot_mapping=scatterplot_mapping,
)

fig.savefig("images/topk1_fulljoin.png")
fig.savefig("images/topk1_fulljoin.pdf")

# %%
