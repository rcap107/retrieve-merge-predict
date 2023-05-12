# %%
# %load_ext autoreload
# %autoreload 2
import json
import pickle
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.candidate_discovery.utils_lazo import LazoIndex
from src.candidate_discovery.utils_minhash import MinHashIndex
from src.data_preparation.data_structures import CandidateJoin, RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins


def prepare_metadata(data_dir, source_data_lake, mdata_out_dir, mdata_index_fname):
    for dataset_path in data_dir.glob("**/*.parquet"):
        ds = RawDataset(dataset_path, source_data_lake, mdata_out_dir)
        ds.save_metadata_to_json()

    metadata_index = MetadataIndex(mdata_out_dir)
    metadata_index.save_index(mdata_index_fname)


def prepare_default_configs(data_dir):
    """Prepare default configurations for various indexing methods and provide the
    data directory that contains the metadata of the tables to be indexed.

    Args:
        data_dir (str): Path to the directory that contains the metadata.

    Raises:
        IOError: Raise IOError if `data_dir` is incorrect.

    Returns:
        dict: Configuration dictionary
    """
    if Path(data_dir).exists():
        configs = {
            "lazo": {
                "data_dir": data_dir,
                "partition_size": 50_000,
                "host": "localhost",
                "port": 15449,
            },
            "minhash": {
                "data_dir": data_dir,
                "thresholds": [20, 40, 80],
                "oneshot": True,
            },
        }
        return configs
    else:
        raise IOError(f"Invalid path {data_dir}")


def prepare_indices(index_configurations: dict):
    """Given a dict of index configurations, initialize the required indices.

    Args:
        index_configurations (dict): Dictionary that contains the required configurations.

    Raises:
        NotImplementedError: Raise NotImplementedError when providing an index that is not recognized.

    Returns:
        dict: Dictionary that contains the initialized indices.
    """
    index_dict = {}
    for index, config in index_configurations.items():
        if index == "lazo":
            index_dict[index] = LazoIndex(**config)
        elif index == "minhash":
            index_dict[index] = MinHashIndex(**config)
        else:
            raise NotImplementedError

    return index_dict


def save_indices(index_dict, index_dir):
    """Save all the indices found in `index_dict` in separate pickle files, in the
    directory provided in `index_dir`.

    Args:
        index_dict (dict): Dictionary containing the indices.
        index_dir (str): Path where the dictionaries will be saved.
    """
    if Path(index_dir).exists():
        for index in index_dict:
            index_name = index.index_name
            filename = f"{index_name}_index.pickle"
            fpath = Path(index_dir, filename)
            with open(fpath, "wb") as fp:
                pickle.dump(index, fp)
    else:
        raise ValueError(f"Invalid `index_dir` {index_dir}")


def load_indices(index_dir):
    """Given `index_dir`, scan the directory and load all the indices in an index dictionary.

    Args:
        index_dir (str): Path to the directory containing the indices.

    Raises:
        IOError: Raise IOError if the index dir does not exist.

    Returns:
        dict: The `index_dict` containing the loaded indices.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise IOError(f"Index dir {index_dir} does not exist.")
    index_dict = {}

    for index_path in index_dir.glob("**/*.pickle"):
        with open(index_path, "rb") as fp:
            index = pickle.load(fp)
            iname = index.index_name
            index_dict[iname] = index

    return index_dict


def generate_candidates(
    index_result: list,
    metadata_index: MetadataIndex,
    mdata_source: dict,
    query_column: str,
):
    """Given the index results in `index_result`, generate a candidate join for each of them. The candidate join will
    not execute the join operation: it holds the information (path, join columns) necessary for it.

    This information is extracted using `metadata_index` for the join columns and `mdata_source` for the source
    table. `query_column` is the join column in the source table.

    Args:
        index_result (list): List of potential join candidates as they were provided by an index.
        metadata_index (MetadataIndex): Metadata Index that holds the metadata of all tables in the data lake.
        mdata_source (dict): Metadata of the source table.
        query_column (str): Label of the query column.

    Returns:
        dict: Dictionary containing the candidates, the candidate id is the index.
    """
    candidates = {}
    for res in index_result:
        hash_, column, similarity = res
        mdata_cand = metadata_index.query_by_hash(hash_)
        cjoin = CandidateJoin(
            source_table_metadata=mdata_source,
            candidate_table_metadata=mdata_cand,
            how="left",
            left_on=query_column,
            right_on=column,
            similarity_score=similarity,
        )
        candidates[cjoin.candidate_id] = cjoin
    return candidates


def querying(
    mdata_source: dict,
    source_column: str,
    query: list,
    indices: dict,
    mdata_index: MetadataIndex,
):
    """Query all indices for the given values in `query`, then generate the join candidates.

    Args:
        mdata_source (dict): Metadata of the source table.
        source_column (str): Column that is the target of the query.
        query (list): List of values in the query column.
        indices (dict): Dictionary containing all the indices.
        mdata_index (MetadataIndex): Metadata Index that contains information on tables in the data lake.

    Returns:
        (dict, dict): Returns the result of the queries on each index and the candidate joins for all results.
    """
    query_results = {}
    for index in indices:
        res = index.query_index(query)
        query_results[index.index_name] = res

    candidates_by_index = {}
    for index, res in query_results.items():
        candidates = generate_candidates(
            index, mdata_index, mdata_source, source_column
        )
        candidates_by_index[index] = candidates

    return query_results, candidates_by_index


def profile_candidates(mdata_source: RawDataset, candidates_by_index: dict):
    """Profile all the join candidates, then return the results in dataframe format.

    Args:
        mdata_source (RawDataset): Metadata of the source table.
        candidates_by_index (dict): Dictionary containing the candidates produced by each index.

    Returns:
        pd.DataFrame: Dataframe containing the profiling results.
    """
    source_table_name = mdata_source.hash

    profiling_results = {}
    for index, candidates in candidates_by_index.items():
        profiling_results[index] = profile_joins(
            mdata_source, source_table_name, candidates
        )

    return profiling_results


def execute_joins():
    pass


# %%
if __name__ == "__main__":
    data_dir = Path("data/yago3-dl/wordnet/subtabs/yago_seltab_wordnet_movie")
    source_data_lake = "yago3-dl"
    output_dir = "data/metadata/debug"
    mdata_index_fname = "debug_metadata_index.pickle"
    index_dir = "data/metadata/indices"

    prepare_indices = True

    prepare_metadata(
        data_dir,
        source_data_lake=source_data_lake,
        mdata_out_dir=output_dir,
        mdata_index_fname=mdata_index_fname,
    )

    if prepare_indices:
        index_configurations = prepare_default_configs()
        indices = prepare_indices(index_configurations)
        save_indices()
    else:
        indices = load_indices()

    query_results, candidates_by_index = querying()

    profile_candidates()


# %%

# %%
# Load prepared indices
mh_index = pickle.load(open("mh_index.pickle", "rb"))
metadata_index = MetadataIndex(index_path="debug_metadata_index.pickle")
# %%
# Query index
data_path = Path("data/yago3-dl/wordnet")
metadata_path = Path("data/metadata/debug")

tab_name = "yago_seltab_wordnet_movie"
tab_path = Path(data_path, f"{tab_name}.parquet")
df = pl.read_parquet(tab_path)
query_column = "isLocatedIn"
query = df[query_column].sample(50000).drop_nulls()

mh_result = mh_index.query_index(query)

# %%
mdata_source = RawDataset(tab_path, "yago3-dl", metadata_path)


# %%
