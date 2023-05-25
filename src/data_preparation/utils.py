import argparse
import json
import pickle
from operator import itemgetter
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl

import src.evaluation.evaluation_methods as em
import src.utils as utils
from src.candidate_discovery.utils_lazo import LazoIndex
from src.candidate_discovery.utils_minhash import MinHashIndex
from src.data_preparation.data_structures import CandidateJoin, RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins


def cast_features(table: pl.DataFrame):
    for col in table.columns:
        try:
            table = table.with_columns(pl.col(col).cast(pl.Float64))
        except pl.ComputeError:
            continue

    cat_features = [k for k, v in table.schema.items() if str(v) == "Utf8"]
    num_features = [k for k, v in table.schema.items() if str(v) == "Float64"]

    return table, num_features, cat_features


class MetadataIndex:
    """This class defines a Metadata Index, which reads the metadata folder to
    have a ready index of the tables found in the data lake. The index is implemented
    as a dictionary where the keys are the hash IDs of the datasets, while the
    metadata of the dataset is the value.
    """

    def __init__(self, metadata_dir=None, index_path=None) -> None:
        """Initialize the class by providing either the metadata directory, or
        the path to a pre-initialized index to load.

        Args:
            metadata_dir (str, optional): Path to the metadata directory. Defaults to None.
            index_path (str, optional): Path to the pre-built index. Defaults to None.

        Raises:
            IOError: Raise IOError if the provided metadata dir does not exist, or isn't a directory.
            ValueError: Raise ValueError if both `metadata_dir` and `index_path` are None.
        """

        if index_path is not None:
            index_path = Path(index_path)
            self.index = self._load_index(index_path)
        elif metadata_dir is not None:
            metadata_dir = Path(metadata_dir)
            if (not metadata_dir.exists()) or (not metadata_dir.is_dir()):
                raise IOError(f"Metadata path invalid.")
            self.index = self.create_index(metadata_dir)
        else:
            raise ValueError("Either `metadata_dir` or `index_path` must be provided.")

    def query_by_hash(self, hashes: Union[list, str]):
        """Given either a hash or a list of hashes, return the metadata of all
        corresponding hashes.

        Args:
            hashes (Union[list, str]): Hash or list of hashes to extract from the index.

        Raises:
            TypeError: Raise TypeError if `hashes` is not a list or a string.

        Returns:
            _type_: The metadata corresponding to the queried hashes.
        """
        # return itemgetter(hashes)(self.index)
        if isinstance(hashes, list):
            return itemgetter(*hashes)(self.index)
        elif isinstance(hashes, str):
            return itemgetter(hashes)(self.index)
        else:
            raise TypeError("Inappropriate type passed to argument `hashes`")

    def create_index(self, metadata_dir):
        """Fill the index dictionary by loading all json files found in the provided directory. The stem of the file name will be used as dictionary key.

        Args:
            metadata_dir (Path): Path to the metadata directory.

        Raises:
            IOError: Raise IOError if the metadata directory is invalid.

        Returns:
            dict: Dictionary containing the metadata, indexed by file hash.
        """
        index = {}
        if metadata_dir.exists() and metadata_dir.is_dir:
            all_files = metadata_dir.glob("**/*.json")
            for fpath in all_files:
                md_hash = fpath.stem
                with open(fpath, "r") as fp:
                    index[md_hash] = json.load(fp)
            return index
        else:
            raise IOError(f"Incorrect path {metadata_dir}")

    def _load_index(self, index_path: Path):
        """Load a pre-built index from the given `index_path`.

        Args:
            index_path (Path): Path to the pre-built index.

        Raises:
            IOError: Raise IOError if the index_path is invalid.

        Returns:
            dict: Dictionary containing the metadata.
        """
        if Path(index_path).exists():
            with open(index_path, "rb") as fp:
                index = pickle.load(fp)
            return index
        else:
            raise IOError(f"Index file {index_path} does not exist.")

    def save_index(self, output_path):
        """Save the index dictionary in the given `output_path` for persistence.

        Args:
            output_path (Path): Output path to use when saving the file.
        """
        with open(output_path, "wb") as fp:
            pickle.dump(self.index, fp)

    def fetch_metadata_by_hash(self, tgt_hash):
        try:
            return self.index[tgt_hash]
        except KeyError:
            raise KeyError(f"Hash {tgt_hash} not found in the index.")


def prepare_default_configs(data_dir, selected_indices=None):
    """Prepare default configurations for various indexing methods and provide the
    data directory that contains the metadata of the tables to be indexed.

    Args:
        data_dir (str): Path to the directory that contains the metadata.
        selected_indices (str): If provided, prepare and run only the selected indices.

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
        if selected_indices is not None:
            return {
                index_name: config
                for index_name, config in configs.items()
                if index_name in selected_indices
            }
        else:
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
        for index_name, index in index_dict.items():
            print(f"Saving index {index_name}")
            filename = f"{index_name}_index.pickle"
            fpath = Path(index_dir, filename)
            index.save_index(fpath)
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
            input_dict = pickle.load(fp)
            iname = input_dict["index_name"]
            if iname == "minhash":
                index = MinHashIndex()
                index.load_index(index_dict=input_dict)
            elif iname == "lazo":
                index = LazoIndex()
                index.load_index(index_path)
            else:
                raise ValueError(f"Unknown index {iname}.")

            index_dict[iname] = index

    return index_dict


def generate_candidates(
    index_name: str,
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
        index_name (str): Name of the index that generated the candidates.
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
            indexing_method=index_name,
            source_table_metadata=mdata_source.info,
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
    for index_name, index in indices.items():
        print(f"Querying index {index_name}.")
        index_res = index.query_index(query)
        query_results[index.index_name] = index_res

    candidates_by_index = {}
    for index, index_res in query_results.items():
        candidates = generate_candidates(
            index, index_res, mdata_index, mdata_source, source_column
        )
        candidates_by_index[index] = candidates

    return query_results, candidates_by_index


def evaluate_joins(source_table, join_candidates: dict, num_features=None, verbose=1):
    result_list = []
    source_table, num_features, cat_features = em.prepare_table_for_evaluation(
        source_table, num_features
    )

    # Run on source table alone
    print("Running on base table.")
    results = em.run_on_table(source_table, num_features, cat_features, verbose=verbose)
    rmse = em.measure_rmse(results["y_test"], results["y_pred"])
    r2 = em.measure_rmse(results["y_test"], results["y_pred"])
    # TODO add proper name + hash
    result_list.append(("base", "base", rmse, r2))

    # Run on all candidates
    print("Running on candidates, one at a time.")
    for index_name, index_cand in join_candidates.items():
        candidate_results_dict = em.execute_on_candidates(
            index_cand, source_table, num_features, cat_features, verbose=verbose
        )
        for k, v in candidate_results_dict.items():
            result_list.append((index_name, k, v))

    # Execute full join by index, then evaluate
    # print("Running on the fully-joined table. ")
    # for index_name, index_cand in join_candidates.items():
    #     merged_table = em.execute_full_join(index_cand, source_table, num_features, cat_features)
    #     cat_features = [col for col in merged_table.columns if col not in num_features]
    #     merged_table = merged_table.fill_null("")
    #     merged_table = merged_table.fill_nan("")
    #     results = em.run_on_table(merged_table, num_features, cat_features, verbose=verbose)
    #     rmse = em.measure_rmse(results["y_test"], results["y_pred"])
    #     result_list.append((index_name, "full_merge", rmse))

    return pl.from_records(result_list, orient="row").to_pandas()
