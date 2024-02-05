#%%
# %cd ~/bench
#%%
from pathlib import Path

import polars as pl
import polars.selectors as cs

import src.utils.plotting as plotting
from src.utils.logging import read_and_process, read_logs

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

# %%
# if __name__ == "__main__":
result_path = "results/overall/overall_aggr.parquet"
df_results = pl.read_parquet(result_path)
results_full, results_depleted = read_and_process(df_results)

case = "dep"
if case == "dep":
    current_results = results_depleted.clone()
    current_results = current_results.filter(pl.col("estimator") != "nojoin")
elif case == "full":
    current_results = results_full.clone()

#%%
var = "aggregation"
scatter_d = "case"
plotting.draw_pair_comparison(
    current_results,
    var,
    form_factor="multi",
    scatterplot_dimension=scatter_d,
    figsize=(10, 2.1),
    scatter_mode="split",
    savefig=True,
    savefig_type=["png", "pdf"],
    case=case,
    colormap_name="Set1",
    jitter_factor=0.01,
    qle=0.05,
    add_titles=False,
)
# %%
