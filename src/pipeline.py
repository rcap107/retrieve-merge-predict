import logging
import os
import pickle
from pathlib import Path

import polars as pl
from joblib import dump, load
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from src.data_structures.indices import LazoIndex, MinHashIndex
from src.data_structures.metadata import CandidateJoin, MetadataIndex
from src.methods import evaluation as em

logger_sh = logging.getLogger("pipeline")


def prepare_dirtree():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/generated_candidates", exist_ok=True)
    os.makedirs("data/metadata/queries", exist_ok=True)


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


def save_indices(index_dict: dict, index_dir: str | Path):
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


def load_index(index_path: str | Path, tab_name=None):
    index_path = Path(index_path)
    if not index_path.exists():
        raise IOError(f"Index {index_path} does not exist.")
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

    return index


def load_indices(index_dir, selected_indices=["minhash"]):
    """Given `index_dir`, scan the directory and load all the indices in an index dictionary.

    Args:
        index_dir (str): Path to the directory containing the indices.
        selected_indices (list, optional): If provided, select only the provided indices.
        tab_name (str, optional): Name of the table, required when using `manual` index.

    Raises:
        IOError: Raise IOError if the index dir does not exist.
        RuntimeError: Raise RuntimeError if `manual` is required as index and `tab_name` is `None`.

    Returns:
        dict: The `index_dict` containing the loaded indices.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise IOError(f"Index dir {index_dir} does not exist.")
    index_dict = {}

    for index_path in index_dir.glob("**/*.pickle"):
        with open(index_path, "rb") as fp:
            input_dict = load(fp)
            iname = input_dict["index_name"]
            if iname in selected_indices:
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


def querying(
    mdata_source: dict,
    query_column: str,
    query: list,
    indices: dict,
    mdata_index: MetadataIndex,
    top_k=15,
):
    """Query all indices for the given values in `query`, then generate the join candidates.

    Args:
        mdata_source (dict): Metadata of the source table.
        query_column (str): Column that is the target of the query.
        query (list): List of values in the query column.
        indices (dict): Dictionary containing all the indices.
        mdata_index (MetadataIndex): Metadata Index that contains information on tables in the data lake.

    Returns:
        (dict, dict): Returns the result of the queries on each index and the candidate joins for all results.
    """
    query_results = {}
    for index_name, index in indices.items():
        # print(f"Querying index {index_name}.")
        index_res = index.query_index(query)
        query_results[index.index_name] = index_res

    candidates_by_index = {}
    for index, index_res in query_results.items():
        candidates = generate_candidates(
            index, index_res, mdata_index, mdata_source, query_column, top_k
        )
        candidates_by_index[index] = candidates

    return query_results, candidates_by_index
