import polars as pl
import polars.selectors as cs
from tqdm import tqdm


def get_cols_by_type(table: pl.DataFrame):
    """Given a dataframe in `table`, find the numeric and string columns, then
    return them in two separate lists.

    Args:
        table (pl.DataFrame): Input table to observe.

    Returns:
        (list, list): A tuple that contains the two lists of numerical and
        categorical columns.
    """
    num_cols = table.select(cs.numeric()).columns
    cat_cols = table.select(cs.string()).columns

    return num_cols, cat_cols


def cast_features(table: pl.DataFrame):
    """Try to cast all columns in a table to float. If the casting operation
    fails, do nothing and keep the previous type.

    Args:
        table (pl.DataFrame): Table to modify.

    Returns:
        pl.DataFrame: Table with updated types.
    """
    for col in table.columns:
        try:
            table = table.with_columns(pl.col(col).cast(pl.Float64))
        except pl.ComputeError:
            table = table.with_columns(pl.col(col).cast(pl.Utf8))
    return table


def prepare_dfs_table(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
):
    """This function takes as input a left and right table to join on, as well
    as the join columns (either `on`, or `left_on` and `right_on`), then uses
    DFS to generate new features. It then returns the table with joined and
    augmented columns.

    Args:
        left_table (pl.DataFrame): Left table to join.
        right_table (pl.DataFrame): Right table to join.
        on (list, optional): List of columns to use for joining. Defaults to None.
        left_on (list, optional): List of columns in the left table to use for joining. Defaults to None.
        right_on (list, optional): List of columns in the right table to use for joining. Defaults to None.

    Raises:
        NotImplementedError: Raise NotImplementedError if more than one column to
        join on is provided.

    Returns:
        pl.DataFrame: The new table.
    """
    # Optimizing imports: these take a long time and are only used when DFS is needed
    import featuretools as ft
    from woodwork.logical_types import Categorical, Double

    def get_logical_types(df):
        num_types = df.select_dtypes("number").columns
        cat_types = [_ for _ in df.columns if _ not in num_types]
        logical_types = {col: Categorical for col in cat_types}
        logical_types.update({col: Double for col in num_types})
        return logical_types

    if on is not None:
        left_on = right_on = on

    if (isinstance(left_on, list) and len(left_on) > 1) or (
        isinstance(right_on, list) and len(right_on) > 1
    ):
        raise NotImplementedError("Many-to-many joins are not supported")
    left_on = left_on[0]
    right_on = right_on[0]

    # DFS does not support joining on columns that include duplicated values.
    left_table_dedup = left_table.unique(left_on).to_pandas()
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

    # `feature_matrix` is the joined table, with new features
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="source_table",
        drop_contains=["target"],
        return_types="all",
        n_jobs=1,
    )
    new_df = feature_matrix.copy()
    cat_cols = new_df.select_dtypes(exclude="number").columns
    num_cols = new_df.select_dtypes("number").columns
    for col in cat_cols:
        new_df[col] = new_df[col].astype(str)
    for col in num_cols:
        new_df[col] = new_df[col].astype(float)

    # The following step is needed to keep the number of samples in `left_table`
    # constant.
    feat_columns = [col for col in new_df.columns if col not in left_table.columns]

    right = pl.from_pandas(new_df[feat_columns].reset_index()).with_columns(
        pl.col(left_on).cast(str)
    )
    augmented_table = left_table.join(right, how="left", on=left_on)

    return augmented_table


def execute_join_with_aggregation(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on: list[str] = None,
    left_on: list[str] = None,
    right_on: list[str] = None,
    how="left",
    aggregation: str = None,
    suffix="_right",
):
    if on is not None:
        if left_on is None and right_on is None:
            left_on = right_on = on
        else:
            raise ValueError(
                "If `on` is provided, `left_on` and `right_on` should be left as None."
            )

    # TODO: this will probably break when right_on contains more than one column, but I don't care for now
    right_table = right_table.filter(pl.col(right_on).is_in(left_table[left_on]))

    if aggregation == "dfs":
        merged = prepare_dfs_table(
            left_table,
            right_table,
            left_on=left_on,
            right_on=right_on,
        )
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
            suffix=suffix,
        )
    return merged


def execute_join_all_candidates(
    source_table: pl.DataFrame, index_cand: dict, aggregation: str
):
    """Execute a full join on `source_table` by chain-joining and aggregating
    all the candidates in `index_cand`, using the aggregation method specified
    in `aggregation`.

    Args:
        source_table (pl.DataFrame): Source table to join on.
        index_cand (dict): Index containing info on the candidate tables.
        aggregation (str): Aggregation function, can be either `first`, `mean`
        or `dfs`.

    Returns:
        pl.DataFrame: The aggregated table.
    """
    merged = source_table.clone()
    hashes = []
    for hash_, mdata in tqdm(
        index_cand.items(),
        total=len(index_cand),
        leave=False,
        desc="Full Join",
        position=2,
    ):
        cnd_md = mdata.candidate_metadata
        hashes.append(cnd_md["hash"])
        candidate_table = pl.read_parquet(cnd_md["full_path"])

        left_on = mdata.left_on
        right_on = mdata.right_on

        if aggregation == "dfs":
            aggr_right = aggregate_table(
                candidate_table,
                right_on,
                aggregation_method=aggregation,
                left_table=source_table,
                left_on=left_on,
            )
        else:
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
    suffix=None,
):
    """Utility function to execute the join between two tables, with error checking.

    Args:
        left_table (pl.DataFrame): Left table to join.
        right_table (pl.DataFrame): Right table to join
        on (list, optional): . Defaults to None.
        left_on (list, optional): List of columns to join on in the left table. Defaults to None.
        right_on (list, optional): List of columns to join on in the right table. Defaults to None.
        how (str, optional): Join strategy. Can be either `left`, `inner`, `outer`. Defaults to "left".
        suffix (str, optional): Suffix to be appended to the columns in the right table in case of overlap. Defaults to None.

    Raises:
        ValueError: Raise ValueError if the provided join strategy is unknown.
        KeyError: Raise KeyError if any join column (in `on`, `left_on`, `right_on`)
                    is not found in the table columns.

    Returns:
        pl.DataFrame: Joined table.
    """
    if suffix is None:
        suffix = ""

    if how not in ["left", "inner", "outer"]:
        raise ValueError(f"Unknown join strategy {how}")

    if on is not None:
        if any(  # if any of the following two conditions is false, raise an exception
            [
                all(
                    c in left_table.columns for c in on
                ),  # all columns in c must be in left_table.columns
                all(
                    c in right_table.columns for c in on
                ),  # all columns in c must be in left_table.columns
            ]
        ):
            raise KeyError("Columns in `on` were not found.")

        joined_table = (
            left_table.lazy().join(right_table.lazy(), on=on, how=how).collect()
        )

    elif left_on is not None and right_on is not None:
        if not all(c in left_table.columns for c in left_on):
            raise KeyError("Not all columns in left_on are found in left_table.")

        if not all(c in right_table.columns for c in right_on):
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
        return joined_table
    else:
        raise ValueError("Both `left_on` and `right_on` are None.")


def aggregate_table(
    right_table, right_on, aggregation_method, left_table=None, left_on=None
):
    """Perfrom aggregation on the given `right_table` by deduplicating rows
    according to the columns specified in `right_on` by using the given
    `aggregation_method`.

    If `aggregation_method` is `dfs`, then `left_table` and `left_on` must also
    be provided.

    Args:
        right_table (Union[pl.DataFrame, pl.LazyFrame]): Table to deduplicate.
        right_on (list): List of columns to deduplicate over.
        aggregation_method (str): Aggregation method to use. Can be one of
        `first`, `mean`, `dfs`.
        left_table (Union[pl.DataFrame, pl.LazyFrame], optional): Left table in
        the join, needed if `aggregation_method` is `dfs`. Defaults to None.
        left_on (list, optional): Columns to deduplicate over in the left table.
        Required if `left_table` is provided. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if aggregation_method == "first":
        aggr_table = aggregate_first(right_table, right_on)
    elif aggregation_method == "mean":
        aggr_table = aggregate_mean(right_table, right_on)
    elif aggregation_method == "dfs":
        if left_table is None or left_on is None:
            raise ValueError(
                "Expected values for `left_table` and `left_on`, "
                f"instead found {left_table} and {left_on}."
            )
        aggr_table = prepare_dfs_table(
            left_table.lazy().collect(),  # needed because `left_table` can be either LazyFrame or DataFrame
            right_table,
            left_on=left_on,
            right_on=right_on,
        )
        to_drop = [
            col
            for col in left_table.columns
            if col not in left_on and col not in right_table.columns
        ]
        aggr_table = aggr_table.drop(to_drop)
        aggr_table = aggr_table.rename(dict(zip(left_on, right_on)))
    else:
        raise ValueError(f"Unknown aggregation method {aggregation_method}")

    return aggr_table


def aggregate_mean(target_table, aggr_columns):
    """Deduplicate all values in `target_table` that are in columns other than
    `aggr_columns`. Values typed as strings are replaced by the mode of each
    group; values typed as numbers are replaced by the mean.

    Args:
        target_table (pl.DataFrame): Table to deduplicate.
        aggr_columns (list): List of columns to group by when deduplicating.

    Returns:
        pl.DataFrame: Deduplicated dataframe.
    """
    df_dedup = target_table.groupby(aggr_columns).agg(
        cs.string().mode().sort(descending=True).first(), cs.numeric().mean()
    )
    return df_dedup


def aggregate_first(target_table: pl.DataFrame, aggr_columns):
    """Deduplicate all values in `target_table` that are in columns other than
    `aggr_columns`. For all duplicated rows, keep only the first occurrence.

    Args:
        target_table (pl.DataFrame): Table to deduplicate.
        aggr_columns (list): List of columns to group by when deduplicating.

    Returns:
        pl.DataFrame: Deduplicated dataframe.
    """

    df_dedup = target_table.unique(aggr_columns, keep="any")
    return df_dedup
