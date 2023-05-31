import polars as pl
from src.utils import data_structures as ds
import pandas as pd
from typing import Iterable

import featuretools as ft
from woodwork.logical_types import Categorical, Double


def get_logical_types(df):
    num_types = df.select_dtypes("number").columns
    cat_types = [_ for _ in df.columns if _ not in num_types]
    logical_types = {col: Categorical for col in cat_types}
    logical_types.update({col: Double for col in num_types})
    return logical_types


def prepare_dfs_table(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
):
    if on is not None:
        left_on = right_on = on

    if (isinstance(left_on, list) and len(left_on) > 1) or (
        isinstance(right_on, list) and len(right_on) > 1
    ):
        raise NotImplementedError("Many-to-many joins are not supported")
    else:
        left_on = left_on[0]
        right_on = right_on[0]

    left_table = left_table.unique("col_to_embed").to_pandas()
    target_column = left_table["target"]
    right_table = right_table.with_row_count("index").to_pandas()

    es = ft.EntitySet()
    left_types = get_logical_types(left_table)
    right_types = get_logical_types(right_table)

    es = es.add_dataframe(
        dataframe_name="source_table",
        dataframe=left_table,
        index=left_on,
        logical_types=left_types,
    )

    es = es.add_dataframe(
        dataframe_name="candidate_table",
        dataframe=right_table,
        index="index",
        logical_types=right_types,
    )

    es = es.add_relationship("source_table", left_on, "candidate_table", right_on)

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="source_table",
        drop_contains=["target"],
        return_types="all",
    )

    feature_matrix["target"] = left_table["target"]

    new_df = feature_matrix.copy().reset_index()
    cat_cols = new_df.select_dtypes(exclude="number").columns
    num_cols = new_df.select_dtypes("number").columns
    for col in cat_cols:
        new_df[col] = new_df[col].astype(str)
    for col in num_cols:
        new_df[col] = new_df[col].astype(float)

    pl_df = pl.from_pandas(new_df)

    return pl_df


def execute_join(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
    dedup=False,
):
    if how not in ["left", "inner", "outer"]:
        raise ValueError(f"Unknown join option {how}")

    if on is not None:
        if on not in left_table.columns or on not in right_table.columns:
            raise ValueError(f"Columns in `on` were not found.")
        else:
            joined_table = (
                left_table.lazy().join(right_table.lazy(), on=on, how=how).collect()
            )

    elif left_on is not None and right_on is not None:
        if not all([c in left_table.columns for c in left_on]):
            raise KeyError("Not all columns in left_on are found in left_table.")

        if not all([c in right_table.columns for c in right_on]):
            raise KeyError("Not all columns in right_on are found in right_table.")

        joined_table = (
            left_table.lazy()
            .join(right_table.lazy(), left_on=left_on, right_on=right_on, how=how)
            .collect()
        )

        if dedup:
            joined_table = prepare_deduplicated_table(joined_table, left_table.columns)

    return joined_table


def prepare_deduplicated_table(table, left_columns):
    df_list = []

    for gkey, group in table.groupby(left_columns):
        g = (
            group.lazy()
            .select(pl.col(left_columns), pl.all().exclude(left_columns).mode().first())
            .collect()
        )
        df_list.append(g)

    df_dedup = pl.concat(df_list)
    return df_dedup
