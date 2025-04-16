# %%
import polars as pl

from src.utils.constants import LABEL_MAPPING

# %%
df_aggregation = pl.read_parquet("results/results_aggregation.parquet")
df_general = pl.read_parquet("results/results_general.parquet")
df_retrieval = pl.read_parquet("results/results_retrieval.parquet")
df_master = pl.read_parquet("results/master_list.parquet")

# %%
variables = ["chosen_model", "jd_method", "estimator", "target_dl", "base_table"]

# %%
df_general.pivot(
    on="estimator",
    index="chosen_model",
    values="prediction_metric",
    aggregate_function="median",
)
# %%
for var_1 in variables:
    df_list = []
    for var_2 in variables:
        if var_1 == var_2:
            continue
        _this_df = df_general.pivot(
            on=var_2,
            index=var_1,
            values="prediction_metric",
            aggregate_function="median",
        )
        _index = _this_df.get_column(var_1).replace(LABEL_MAPPING[var_1])
        _this_df.drop_in_place(var_1)
        _this_df = _this_df.rename(lambda c : LABEL_MAPPING[var_2][c])
        _col_order = [var_1] + _this_df.columns
        _this_df = _this_df.with_columns(_index.alias(var_1)).select(_col_order)
        df_list.append(_this_df)

    df_aligned = pl.concat(df_list, how="align")
    df_aligned.write_csv(f"results/results_pivot_{var_1}.csv")

# %%
# %%
