import polars as pl
from src.data_preparation import data_structures as ds
import pandas as pd
from typing import Iterable, Union, List
from pathlib import Path


def execute_dummy_join(
    left_table: pl.DataFrame,
    right_table: pl.DataFrame,
    on=None,
    left_on=None,
    right_on=None,
    how="left",
):
    if how not in ["left", "inner", "outer"]:
        raise ValueError(f"Unknown join option {how}")

    if on is not None:
        if on not in left_table.columns or on not in right_table.columns:
            raise ValueError(f"Columns in `on` were not found.")
        else:
            left_table = left_table[on]
            right_table = right_table[on]

            joined_table = (
                left_table.lazy().join(right_table.lazy(), on=on, how=how).collect()
            )

    elif left_on is not None and right_on is not None:
        left_table = left_table[on]
        right_table = right_table[on]

        joined_table = (
            left_table.lazy()
            .join(right_table.lazy(), left_on=left_on, right_on=right_on, how=how)
            .collect()
        )

    return joined_table


def get_unique_keys(table, columns):
    uk = find_unique_keys(table, columns)
    if uk is not None:
        return len(uk)
    else:
        return None


def find_unique_keys(df, key_cols):
    """Find the set of unique keys given a combination of columns.

    This function is used to find what is the potential cardinality of a join key.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Dataframe to estimate key cardinality on.
        key_cols (list): List of key columns.

    Returns:
        _type_: List of unique keys.
    """
    try:
        unique_keys = df[key_cols].unique()
    except pl.DuplicateError:
        unique_keys = None

    return unique_keys


def prepare_base_dict_info():
    d_info = {
        "ds_name": None,
        "candidate_name": None,
        "left_on": None,
        "right_on": None,
        "merged_rows": None,
        "scale_factor": None,
        "left_unique_keys": None,
        "right_unique_keys": None,
    }
    return d_info


def merge_table(
    left_table: Union[pd.DataFrame, pl.DataFrame],
    right_table: Union[pd.DataFrame, pl.DataFrame],
    left_on: list,
    right_on: list,
    how: str = "left",
):
    """Merge tables according to the specified engine.

    Args:
        left_table (Union[pd.DataFrame, pl.DataFrame]): Left table to be joined.
        right_table (Union[pd.DataFrame, pl.DataFrame]): Right table to be joined.
        left_on (list): Join keys in the left table.
        right_on (list): Join keys in the right table.
        how (str, optional): Join type. Defaults to "left".

    Raises:
        ValueError: Raises ValueError if the engine provided is not in [`polars`, `pandas`].

    Returns:
        pl.DataFrame: Merged table.
    """
    merged = (
        left_table[left_on]
        .lazy()
        .join(right_table[right_on].lazy(), left_on=left_on, right_on=right_on, how=how)
    )
    return merged.collect()


def measure_containment(
    unique_source: pl.DataFrame,
    unique_cand: pl.DataFrame,
    left_on,
    right_on
):
    # s1 = set(unique_source[left_on].to_series().to_list())
    # s2 = set(unique_cand[right_on].to_series().to_list())
    # return len(s1.intersection(s2))/len(unique_source)
    intersection=unique_source.join(unique_cand, left_on=left_on, right_on=right_on, how="inner")
    return len(intersection) / len(unique_source)


def measure_cardinality_proportion(
    source_table: pl.DataFrame,
    candidate_table: pl.DataFrame,
):
    cardinality_source = len(source_table)
    cardinality_cand = len(candidate_table)

    c1 = cardinality_source / cardinality_cand
    
    return c1


def measure_join_quality(
    source_table,
    candidate_table,
    left_on,
    right_on,
    constants={
        "C_H": 0.75,
        "C_G": 0.5,
        "C_M": 0.25,
        "C_L": 0.1,
        "K_H": 0.25,
        "K_G": 0.125,
        "K_M": 1 / 12,
    }
):
    unique_source = find_unique_keys(source_table, left_on)
    unique_cand = find_unique_keys(candidate_table, right_on)
    containment = measure_containment(unique_source, unique_cand, left_on, right_on)
    card_prop = measure_cardinality_proportion(
        source_table, candidate_table
    )

    if containment >= constants["C_H"] and card_prop >= constants["K_H"]:
        return 4

    elif containment >= constants["C_G"] and card_prop >= constants["K_G"]:
        return 3

    elif containment >= constants["C_M"] and card_prop >= constants["K_M"]:
        return 2

    elif containment >= constants["C_P"]:
        return 1

    else:
        return 0


def profile_joins(
    source_table: pl.DataFrame, source_table_name: str, join_candidates: List, verbose=0
):
    """This function takes as input a source table and a list of join candidates,
    then runs a series of profiling operation on them.

    It notes down `ds_name`, `candidate_name`, `left_on`, `right_on`, size of the
    left join in `merged_rows`, `scale_factor`, cardinality of left and right
    key columns in `left_unique_keys` and `right_unique_keys`.

    Args:

        verbose (int, optional): How much information on dataset failures to be printed. Defaults to 0.

    Returns:
        pd.DataFrame: Summary of the statistics in DataFrame format.
    """

    left_table = source_table.copy()
    all_info_dict = {}
    idx_info = 0

    # TODO: I copypasted, it needs to be adapted to the format that is being spat out by the queries
    for cand_id, join_column, threshold in join_candidates:
        right_table_path = Path(cand_path, cand_id, "tables/learningData.csv")
        base_dict_info = dict(prepare_base_dict_info())
        base_dict_info["ds_name"] = ds_name
        base_dict_info["candidate_name"] = cand_id
        if not right_table_path.exists():
            if verbose > 0:
                print(f"{right_table_path} not found.")
            dict_info = dict(base_dict_info)
            all_info_dict[idx_info] = dict_info
            idx_info += 1
        else:
            right_table = read_table_csv(right_table_path, engine)
            for join_cand in cand_info["metadata"]:
                left_on = join_cand["left_columns_names"][0]
                right_on = join_cand["right_columns_names"][0]
                dict_info = dict(base_dict_info)
                if len(right_on) != len(left_on):
                    if verbose > 1:
                        print(f"Left: {left_on} != Right: {right_on}")
                elif any(r not in right_table.columns for r in right_on) or any(
                    l not in left_table.columns for l in left_on
                ):
                    if verbose > 0:
                        print(f"Not all columns found.")
                else:
                    dict_info["left_rows"], dict_info["left_cols"] = left_table.shape

                    merged = merge_table(
                        left_table=left_table,
                        right_table=right_table,
                        left_on=left_on,
                        right_on=right_on,
                        engine=engine,
                    )

                    dict_info["left_on"] = left_on
                    dict_info["right_on"] = right_on
                    dict_info["merged_rows"] = len(merged)
                    dict_info["scale_factor"] = len(merged) / dict_info["left_rows"]
                    dict_info["left_unique_keys"] = get_unique_keys(left_table, left_on)
                    dict_info["right_unique_keys"] = get_unique_keys(
                        right_table, right_on
                    )

                all_info_dict[idx_info] = dict_info
                idx_info += 1
        df_info = pd.DataFrame().from_dict(all_info_dict, orient="index")
    else:
        df_info = pd.DataFrame()
    return df_info
