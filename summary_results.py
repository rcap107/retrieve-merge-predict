# %%
# %cd ~/bench
# %%
import polars as pl

from src.utils.constants import GROUPING_KEYS

keys = [_ for _ in GROUPING_KEYS if _ != "fold_id"]
# %%
df_aggregation = pl.read_parquet("results/results_aggregation.parquet")
df_general = pl.read_parquet("results/results_general.parquet")
df_retrieval = pl.read_parquet("results/results_retrieval.parquet")
# %%
df_master = pl.read_parquet("results/master_list.parquet")

# %%
time_ = (
    df_master.group_by(GROUPING_KEYS)
    .agg(pl.col("time_run").mean())
    .group_by("chosen_model")
    .agg(pl.col("time_run").sum())
)
# %%
time_.write_csv("results/total_time_final.csv")
# %%
cutby = 0.2

_d = (
    df_general.group_by(GROUPING_KEYS)
    .agg(pl.col("prediction_metric").last())
    .group_by(["base_table", "target_dl"])
    .agg(pl.col("prediction_metric").sort())
    .with_columns(len_l=pl.col("prediction_metric").list.len())
    .with_columns(
        start=(pl.col("len_l") * cutby).cast(pl.Int32),
        end=pl.col("len_l") - (pl.col("len_l") * cutby * 2).cast(pl.Int32),
    )
    .with_columns(
        trimmed=(pl.col("prediction_metric").list.slice(pl.col("start"), pl.col("end")))
    )
    .with_columns(
        trimmed_mean=pl.col("trimmed").list.mean(),
        trimmed_std=pl.col("trimmed").list.std(),
    )
)

# %%
_d.select("base_table", "target_dl", "trimmed_mean", "trimmed_std").pivot(
    values=["trimmed_mean", "trimmed_std"], index="base_table", columns="target_dl"
).sort("base_table").write_csv("trimmed_mean.csv")
# %%
dedup = df_master.group_by(GROUPING_KEYS).agg(
    pl.col("prediction_metric").mean(), pl.col("time_run").mean()
)
# %%
reference_config = {
    "jd_method": "exact_matching",
    "estimator": "best_single_join",
    "chosen_model": "catboost",
    "aggregation": "first",
}
all_variables = [k for k in reference_config.keys()]

df_best = dedup.filter(**reference_config)

df_reference = df_master.filter(
    (~pl.col("estimator").is_in(["nojoin", "top_k_full_join"]))
    & (pl.col("chosen_model") != "linear")
    & (
        pl.col("source_table").is_in(
            [
                "company_employees",
                "housing_prices",
                "us_accidents_2021",
                # "us_accidents_large",
                "us_county_population",
                "us_elections",
                "schools",
            ]
        )
    )
)


# %%
# retrieval method
target = "jd_method"
this_key = [_ for _ in GROUPING_KEYS if _ != target]
_1 = (
    df_reference.join(df_best, on=this_key)
    .with_columns(
        diff_metric=pl.col("prediction_metric") - pl.col("prediction_metric_right"),
        diff_time=pl.col("time_run") / pl.col("time_run_right"),
    )
    .select(GROUPING_KEYS + ["diff_metric", "diff_time"])
    .group_by(all_variables)
    .agg(pl.median("diff_metric") * 100, pl.median("diff_time"))
    .sort("diff_metric", descending=True)
)

# %%
# selector
target = "estimator"
this_key = [_ for _ in GROUPING_KEYS if _ != target]
_2 = (
    df_reference.join(df_best, on=this_key)
    .with_columns(
        diff_metric=pl.col("prediction_metric") - pl.col("prediction_metric_right"),
        diff_time=pl.col("time_run") / pl.col("time_run_right"),
    )
    .select(GROUPING_KEYS + ["diff_metric", "diff_time"])
    .group_by(all_variables)
    .agg(pl.median("diff_metric") * 100, pl.median("diff_time"))
    .sort("diff_metric", descending=True)
)

# %%
# ml model
target = "chosen_model"
this_key = [_ for _ in GROUPING_KEYS if _ != target]
_3 = (
    df_reference.join(df_best, on=this_key)
    .with_columns(
        diff_metric=pl.col("prediction_metric") - pl.col("prediction_metric_right"),
        diff_time=pl.col("time_run") / pl.col("time_run_right"),
    )
    .select(GROUPING_KEYS + ["diff_metric", "diff_time"])
    .group_by(all_variables)
    .agg(pl.median("diff_metric") * 100, pl.median("diff_time"))
    .sort("diff_metric", descending=True)
)


# %%
# aggregation
target = "aggregation"
this_key = [_ for _ in GROUPING_KEYS if _ != target]
_4 = (
    df_reference.join(df_best, on=this_key)
    .with_columns(
        diff_metric=pl.col("prediction_metric") - pl.col("prediction_metric_right"),
        diff_time=pl.col("time_run") / pl.col("time_run_right"),
    )
    .select(GROUPING_KEYS + ["diff_metric", "diff_time"])
    .group_by(all_variables)
    .agg(pl.median("diff_metric") * 100, pl.median("diff_time"))
    .sort("diff_metric", descending=True)
)

# %%
_1.write_csv("results/diff_from_ref/retrieval_method.csv")
_2.write_csv("results/diff_from_ref/selector.csv")
_3.write_csv("results/diff_from_ref/ml_model.csv")
_4.write_csv("results/diff_from_ref/aggregation.csv")

# %%
