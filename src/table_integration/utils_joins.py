import polars as pl
from src.utils import data_structures as ds
import pandas as pd
from typing import Iterable


def execute_join(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
    dedup=False
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
        
        if not all(
                [c in left_table.columns for c in left_on]
            ):
            raise KeyError("Not all columns in left_on are found in left_table.")
        
        if not all (
                [c in right_table.columns for c in right_on]
            ):
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
        g = group.lazy().select(pl.col(left_columns),
        pl.all().exclude(left_columns).mode().first()
        ).collect()
        df_list.append(g)

    df_dedup = pl.concat(df_list)
    return df_dedup
