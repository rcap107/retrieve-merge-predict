# # %%
# %cd ~/bench

# # %%
# %load_ext autoreload
# %autoreload 2
import polars as pl
from memory_profiler import memory_usage

# %%
from src.data_structures.retrieval_methods import ExactMatchingIndex, MinHashIndex

# %%
config_exact = {
    "metadata_dir": "data/metadata/binary_update",
    "base_table_path": "data/source_tables/yadl/us_elections_dems-yadl-depleted.parquet",
    "query_column": "col_to_embed",
    "n_jobs": 1,
}

config_minhash = {
    "metadata_dir": "data/metadata/binary_update",
    "n_jobs": 2,
}

# %%
index = ExactMatchingIndex(**config_exact)
# %%
index = MinHashIndex(**config_minhash)

# %%
# index.query_index()

# %%
cd = pl.read_parquet("data/yadl/binary_update/binary-hasLongitude.parquet")
base_table = pl.read_parquet(
    "data/source_tables/yadl/us_elections_dems-yadl-depleted.parquet"
)

# %%


# %%
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
        # Raise exception if a column name is duplicated
        unique_keys = None

    return unique_keys


# %%
def measure_containment(unique_base_table, candidate_table: pl.DataFrame, right_on):
    unique_cand = find_unique_keys(candidate_table, right_on)

    s1 = unique_base_table
    s2 = set(unique_cand[right_on].to_series().to_list())
    return len(s1.intersection(s2)) / len(s1)


# %%
def measure_containment_new(unique_base_table, candidate_table: pl.DataFrame, right_on):

    return len(
        unique_base_table.implode()
        .list.set_intersection(
            candidate_table.select(pl.col(right_on).unique()).to_series().implode()
        )
        .explode()
    ) / len(unique_base_table)
    # b = candidate_table.select(pl.col(right_on).unique()).to_series().implode()
    # return len(unique_base_table.implode().list.set_intersection(b).explode())/len(unique_base_table)
