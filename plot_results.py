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
    root_path = Path("results/big_batch")
    df_list = []
    for rpath in root_path.iterdir():
        df_raw = read_logs(exp_name=None, exp_path=rpath)
        df_list.append(df_raw)

    df_results = pl.concat(df_list)

    results_full, results_depleted = read_and_process(df_results)

    case = "full"

    if case == "dep":
        current_results = results_depleted.clone()
        current_results = current_results.filter(pl.col("estimator") != "nojoin")

    elif case == "full":
        current_results = results_full.clone()

    for var in ["jd_method"]:
        # for var in ["estimator", "jd_method", "chosen_model"]:
        print(f"Variable: {var}")
        for scatter_d in ["base_table", "chosen_model", "jd_method", "estimator"]:
            if scatter_d == var:
                continue
            plotting.draw_triple_comparison(
                current_results,
                var,
                scatterplot_dimension=scatter_d,
                figsize=(18, 5),
                scatter_mode="split",
                savefig=True,
                case=case,
            )
