import polars as pl
import pandas as pd

import featuretools as ft
from woodwork.logical_types import Categorical, Double
from tqdm import tqdm


def get_logical_types(df):
    num_types = df.select_dtypes("number").columns
    cat_types = [_ for _ in df.columns if _ not in num_types]
    logical_types = {col: Categorical for col in cat_types}
    logical_types.update({col: Double for col in num_types})
    return logical_types


def cast_features(table: pl.DataFrame, only_types=False):
    if not only_types:
        for col in table.columns:
            try:
                table = table.with_columns(pl.col(col).cast(pl.Float64))
            except pl.ComputeError:
                continue

    cat_features = [k for k, v in table.schema.items() if str(v) == "Utf8"]
    num_features = [k for k, v in table.schema.items() if str(v) == "Float64"]

    return table, num_features, cat_features


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
        new_df[feat_columns].reset_index(), how="left", on="col_to_embed"
    )

    pl_df = pl.from_pandas(augmented_table)

    return pl_df


def execute_join_with_aggregation(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
    aggregation=None,
    suffix=None,
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
        aggr_right = aggregate_table(
            right_table, right_on, aggregation_method=aggregation
        )

        merged = execute_join(
            left_table,
            aggr_right,
            left_on=left_on,
            right_on=right_on,
            how=how,
            # mean=dedup,
        )
    return merged


def execute_join_all_candidates(source_table, index_cand, aggregation):
    merged = source_table.clone()
    hashes = []
    for hash_, mdata in tqdm(index_cand.items(), total=len(index_cand)):
        cnd_md = mdata.candidate_metadata
        hashes.append(cnd_md["hash"])
        candidate_table = pl.read_parquet(cnd_md["full_path"])

        left_on = mdata.left_on
        right_on = mdata.right_on

        aggr_right = aggregate_table(
            candidate_table, right_on, aggregation_method=aggregation
        )

        merged = execute_join(
            merged,
            aggr_right,
            left_on=left_on,
            right_on=right_on,
            how="left",
            suffix="_" + hash_[:10],
        )

    return merged


def execute_join(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
    mean=None,
    suffix=None,
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
            .join(
                right_table.lazy(),
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffix=suffix,
            )
            .collect()
        )

        if mean:
            joined_table = aggregate_mean(joined_table, left_table.columns)
        else:
            joined_table = aggregate_first(joined_table, left_table.columns)

    return joined_table


def aggregate_table(target_table, aggr_columns, aggregation_method):
    if aggregation_method == "first":
        aggr_table = aggregate_first(target_table, aggr_columns)
    elif aggregation_method == "mean":
        aggr_table = aggregate_mean(target_table, aggr_columns)
    elif aggregation_method == "dfs":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown aggregation method {aggregation_method}")

    return aggr_table


def aggregate_mean(target_table, target_columns):
    cat_cols = [col for col, dtype in target_table.schema.items() if dtype == pl.Utf8]
    num_cols = [col for col in target_table.columns if col not in cat_cols]

    df_list = []

    for _, group in target_table.groupby(target_columns):
        g = (
            group.lazy()
            .select(
                pl.col(cat_cols).mode().first(),
                pl.col(num_cols).mean(),
            )
            .collect()
        )
        df_list.append(g)

    df_dedup = pl.concat(df_list).unique()
    return df_dedup


def aggregate_first(target_table: pl.DataFrame, aggr_columns):
    df_dedup = target_table.unique(aggr_columns, keep="first")
    return df_dedup
