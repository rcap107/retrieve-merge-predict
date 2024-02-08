from pathlib import Path

import polars as pl
import polars.selectors as cs

import src.utils.plotting as plotting
from src.utils.logging import read_and_process, read_logs

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


if __name__ == "__main__":
    result_path = "results/overall/overall_first.parquet"
    df_results = pl.read_parquet(result_path)
    results_full, results_depleted = read_and_process(df_results)

    case = "dep"
    if case == "dep":
        current_results = results_depleted.clone()
        current_results = current_results.filter(pl.col("estimator") != "nojoin")
    elif case == "full":
        current_results = results_full.clone()

    # for var in ["estimator", "jd_method", "chosen_model"]:
    #     print(f"Variable: {var}")
    #     n_unique = current_results.select(pl.col(var).n_unique()).item()
    #     for scatter_d in ["case"]:
    #         if scatter_d == var:
    #             continue
    #         form_factor = "binary" if n_unique == 2 else "multi"
    #         if form_factor == "multi":
    #             figsize = (12, 3)
    #         else:
    #             figsize = (8, 1)
    #         plotting.draw_pair_comparison(
    #             current_results,
    #             var,
    #             form_factor=form_factor,
    #             scatterplot_dimension=scatter_d,
    #             figsize=figsize,
    #             scatter_mode="split",
    #             savefig=True,
    #             savefig_type=["png", "pdf"],
    #             savefig_tag="open_data",
    #             case=case,
    #             colormap_name="Set1",
    #             qle=0.01,
    #             jitter_factor=0.02
    #         )

    # Plot retrieval method
    var = "jd_method"
    scatter_d = "case"
    plotting.draw_pair_comparison(
        current_results,
        var,
        form_factor="multi",
        scatterplot_dimension=scatter_d,
        figsize=(10, 2.5),
        scatter_mode="split",
        savefig=True,
        savefig_type=["png", "pdf"],
        case=case,
        colormap_name="Set1",
        jitter_factor=0.01,
        qle=0.05,
        add_titles=True,
    )

    # Plot estimator
    var = "estimator"
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
        qle=0.005,
        add_titles=False,
    )

    # Plot ML model
    var = "chosen_model"
    scatter_d = "case"
    plotting.draw_pair_comparison(
        current_results,
        var,
        form_factor="multi",
        scatterplot_dimension=scatter_d,
        figsize=(8, 1),
        scatter_mode="split",
        savefig=True,
        savefig_type=["png", "pdf"],
        case=case,
        colormap_name="Set1",
        jitter_factor=0.01,
        qle=0.05,
        add_titles=False,
    )
