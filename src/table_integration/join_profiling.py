import polars as pl
from src.utils import data_structures as ds
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
        .drop_nulls()
        .join(
            right_table[right_on].lazy().drop_nulls(),
            left_on=left_on,
            right_on=right_on,
            how=how,
        )
    )
    return merged.collect()


def estimate_join_size(source_table, candidate_table, left_on, right_on):
    unique_source = find_unique_keys(source_table, left_on)
    unique_cand = find_unique_keys(candidate_table, right_on)

    numerator = len(source_table) * len(candidate_table)

    estimates = [numerator / len(unique_source), numerator / len(unique_cand)]

    return estimates


def measure_containment(
    source_table: pl.DataFrame, candidate_table: pl.DataFrame, left_on, right_on
):
    unique_source = find_unique_keys(source_table, left_on)
    unique_cand = find_unique_keys(candidate_table, right_on)

    s1 = set(unique_source[left_on].to_series().to_list())
    s2 = set(unique_cand[right_on].to_series().to_list())
    return len(s1.intersection(s2)) / len(s1)
    intersection = unique_source.join(
        unique_cand, left_on=left_on, right_on=right_on, how="inner"
    )
    return len(intersection) / len(unique_source)


def measure_cardinality_proportion(
    source_table: pl.DataFrame, candidate_table: pl.DataFrame, left_on, right_on
):
    unique_source = find_unique_keys(source_table, left_on)
    unique_cand = find_unique_keys(candidate_table, right_on)

    cardinality_source = len(unique_source)
    cardinality_cand = len(unique_cand)

    card_prop = cardinality_source / cardinality_cand

    return card_prop


def measure_join_quality(
    source_table,
    candidate_table,
    left_on,
    right_on,
    constants={
        "C_H": 0.75,
        "C_G": 0.5,
        "C_M": 0.25,
        "C_P": 0.1,
        "K_H": 0.25,
        "K_G": 0.125,
        "K_M": 1 / 12,
    },
):
    unique_source = find_unique_keys(source_table, left_on)
    unique_cand = find_unique_keys(candidate_table, right_on)
    containment = measure_containment(unique_source, unique_cand, left_on, right_on)
    card_prop = measure_cardinality_proportion(
        source_table, candidate_table, left_on, right_on
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


def profile_joins(join_candidates: dict, logger):
    
    tot_dict = []
    for index_name, candidates in join_candidates.items():

        for hash_, mdata in candidates.items():
            prof_dict = {}
            source_metadata = mdata.source_metadata
            candidate_metadata = mdata.candidate_metadata
            source_table = pl.read_parquet(source_metadata["full_path"])
            candidate_table = pl.read_parquet(candidate_metadata["full_path"])
            left_on = mdata.left_on
            right_on = mdata.right_on
            estimates = estimate_join_size(
                source_table, candidate_table, left_on, right_on
            )
            join_quality = measure_join_quality(
                source_table, candidate_table, left_on, right_on
            )
            containment = measure_containment(
                source_table, candidate_table, left_on, right_on
            )
            card_prop = measure_cardinality_proportion(
                source_table, candidate_table, left_on, right_on
            )
            if min(estimates) < 1e6:
                joined = merge_table(
                    source_table,
                    candidate_table,
                    left_on=left_on,
                    right_on=right_on,
                    how="inner",
                )
            else:
                joined = None
            prof_dict["candidate_id"] = mdata.candidate_id
            prof_dict["index"] = index_name
            prof_dict["source_table"] = source_metadata["full_path"]
            prof_dict["source_name"] = source_metadata["df_name"]
            prof_dict["candidate_table"] = candidate_metadata["full_path"]
            prof_dict["candidate_name"] = candidate_metadata["df_name"]
            prof_dict["min_estimate"] = min(estimates)
            prof_dict["join_quality"] = join_quality
            prof_dict["containment"] = containment
            prof_dict["cardinality_proportion"] = card_prop
            if joined is not None:
                prof_dict["true_join_size"] = len(joined)
            else:
                prof_dict["true_join_size"] = 0

            tot_dict.append(prof_dict)

    logger.add_dict("profiling_results", tot_dict)
    
    prof_df = pl.from_dicts(tot_dict)
    return prof_df.to_pandas()
