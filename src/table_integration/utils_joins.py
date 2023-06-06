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

    left_table_dedup = left_table.unique("col_to_embed").to_pandas()
    target_column = left_table["target"]
    right_table = right_table.with_row_count("index").to_pandas()

    es = ft.EntitySet()
    left_types = get_logical_types(left_table_dedup)
    right_types = get_logical_types(right_table)

    es = es.add_dataframe(
        dataframe_name="source_table",
        dataframe=left_table_dedup,
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

    feature_matrix["target"] = left_table_dedup["target"]

    new_df = feature_matrix.copy()
    cat_cols = new_df.select_dtypes(exclude="number").columns
    num_cols = new_df.select_dtypes("number").columns
    for col in cat_cols:
        new_df[col] = new_df[col].astype(str)
    for col in num_cols:
        new_df[col] = new_df[col].astype(float)

    feat_columns = [col for col in new_df.columns if col not in left_table.columns]
    augmented_table = left_table.to_pandas().merge(
        new_df[feat_columns].reset_index(),
        how="left",
        on="col_to_embed"
    )

    pl_df = pl.from_pandas(augmented_table)

    return pl_df


def execute_join_complete(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
    aggregation=None,
    suffix=None  
):
    if aggregation == "dfs":
        # logger.debug("Start DFS.")

        merged = prepare_dfs_table(
            left_table,
            right_table,
            left_on=left_on,
            right_on=right_on,
        )
        # logger.debug("End DFS.")

    else:
        if aggregation == "dedup":
            dedup = True
            jstr = how + "_dedup"
        else:
            jstr = how + "_none"
            dedup = False

        merged = execute_join(
            left_table,
            right_table,
            left_on=left_on,
            right_on=right_on,
            how=how,
            dedup=dedup,
        )
    return merged



def execute_join(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
    dedup=None,
    suffix=None
):
    if suffix is None:
        suffix = ""
    
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
            .join(right_table.lazy(), left_on=left_on, right_on=right_on, how=how, suffix=suffix)
            .collect()
        )

        if dedup:
            joined_table = prepare_deduplicated_table(joined_table, left_table.columns)

    return joined_table


def prepare_deduplicated_table(joined_table, left_columns):
    df_list = []

    for gkey, group in joined_table.groupby(left_columns):
        g = (
            group.lazy()
            .select(pl.col(left_columns), pl.all().exclude(left_columns).mode().first())
            .collect()
        )
        df_list.append(g)

    df_dedup = pl.concat(df_list).unique()
    return df_dedup
