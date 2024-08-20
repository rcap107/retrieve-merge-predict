"""
Main script for figure 3: plot the comparison between different methods.
"""
#%%
import polars as pl

from src.utils import plotting, constants
from src.utils.logging import read_and_process

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)


def get_cases(df: pl.DataFrame, keep_nojoin: bool = False) -> dict:
    if not keep_nojoin:
        df = df.filter(pl.col("estimator") != "nojoin")

    cases = (
        df.select(pl.col(["jd_method", "chosen_model", "estimator"]).unique().implode())
        .transpose(include_header=True)
        .to_dict(as_series=False)
    )

    return dict(zip(*list(cases.values())))

#%%
def prepare_general():
    result_path = "stats/overall/overall_first.parquet"

    # Use the standard method for reading all results for consistency.
    df_results = pl.read_parquet(result_path)
    current_results = read_and_process(df_results)
    others = [col for col in current_results.columns if col not in constants.GROUPING_KEYS + ["case"]]
    current_results = current_results.group_by(constants.GROUPING_KEYS + ["case"]).agg(
    pl.mean(others)
    )
    current_results = current_results.filter(pl.col("estimator") != "nojoin")
    _d = current_results.filter(
        (pl.col("estimator") != "top_k_full_join") & (pl.col("jd_method") != "starmie")
    )
    return _d


#%%

save_figures = True

current_results = prepare_general()
case="dep"

# Plot estimator
var = "estimator"
scatter_d = "case"
plotting.draw_pair_comparison(
    current_results,
    var,
    scatterplot_dimension=scatter_d,
    figsize=(10, 2.1),
    scatter_mode="split",
    savefig=save_figures,
    savefig_type=["png", "pdf"],
    case=case,
    colormap_name="Set1",
    jitter_factor=0.01,
    qle=0.005,
    add_titles=True,
)

# Plot ML model
var = "chosen_model"
scatter_d = "case"
plotting.draw_pair_comparison(
    current_results,
    var,
    scatterplot_dimension=scatter_d,
    figsize=(8, 1),
    scatter_mode="split",
    savefig=save_figures,
    savefig_type=["png", "pdf"],
    case=case,
    colormap_name="Set1",
    jitter_factor=0.01,
    qle=0.05,
    add_titles=False,
)

# %%
