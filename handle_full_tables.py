# %%
import polars as pl
from pathlib import Path
from src.utils import constants


# %%
def get_run_df(run_path, exp_task="regression"):
    dfs = []

    scenario_id = sum(1 for _ in Path(run_path, "run_logs").iterdir())
    for _i in range(scenario_id):
        json_path = Path(run_path, "json", str(_i) + ".json")
        log_path = Path(run_path, "run_logs", str(_i) + ".log")
        _df = pl.read_csv(log_path)
        if exp_task == "regression":
            _df = _df.with_columns(
                pl.lit(0.0).alias("auc"),
                pl.lit(0.0).alias("f1score"),
                prediction_metric=pl.col("r2score"),
            )
        else:
            _df = _df.with_columns(
                pl.lit(0.0).alias("r2score"),
                pl.lit(0.0).alias("rmse"),
                prediction_metric=pl.col("f1score"),
            )

        dfs.append(_df)
    dd = pl.concat(dfs)

    dd = dd.with_columns(
        case=(
            pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
        ),
        y=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score")),
    )

    return dd


projection = [
    "fold_id",
    "target_dl",
    "jd_method",
    "base_table",
    "query_column",
    "estimator",
    "chosen_model",
    "aggregation",
    "y",
    "time_fit",
    "time_predict",
    "time_run",
    "peak_fit",
    "peak_predict",
]


all_dfs = []
for f in Path("results/logs/full_tables").iterdir():
    x = get_run_df(f).select(projection)
    all_dfs.append(x)
df = pl.concat(all_dfs)

# %%
df.pivot(on="base_table", index="estimator", values="y", aggregate_function="mean")
# %%
df_base = pl.read_csv("results/results_general.csv")
# %%
df_base.filter(
    (
        pl.col("estimator").is_in(["best_single_join", "nojoin", "full_join"])
        & (pl.col("chosen_model") == "catboost")
        & (~pl.col("target_dl").is_in(["open_data_us"]))
    )
).pivot(
    on="base_table",
    index="estimator",
    values="prediction_metric",
    aggregate_function="mean",
)

# %%
