import pickle
from pathlib import Path

import polars as pl

import src.evaluation.evaluation_methods as em
from src.candidate_discovery.utils_lazo import LazoIndex
from src.candidate_discovery.utils_minhash import MinHashIndex
from src.utils.data_structures import CandidateJoin, RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins
from src.utils.data_structures import RunLogger

import git

import cProfile, pstats, io
from pstats import SortKey


repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha


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
                "thresholds": [10, 20, 80],
                "oneshot": True,
                "num_perm": 128,
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
    top_k=15,
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

    if top_k > 0:
        # TODO rewrite this so it's cleaner
        ranking = [(k, v.similarity_score) for k, v in candidates.items()]
        clamped = [x[0] for x in sorted(ranking, key=lambda x: x[1], reverse=True)][
            :top_k
        ]

        candidates = {k: v for k, v in candidates.items() if k in clamped}
    return candidates


def querying(
    mdata_source: dict,
    source_column: str,
    query: list,
    indices: dict,
    mdata_index: MetadataIndex,
    top_k=15,
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
        # print(f"Querying index {index_name}.")
        index_res = index.query_index(query)
        query_results[index.index_name] = index_res

    candidates_by_index = {}
    for index, index_res in query_results.items():
        candidates = generate_candidates(
            index, index_res, mdata_index, mdata_source, source_column, top_k
        )
        candidates_by_index[index] = candidates

    return query_results, candidates_by_index


def evaluate_joins(
    source_table,
    source_metadata,
    scenario_logger,
    join_candidates: dict,
    num_features=None,
    verbose=1,
    iterations=1000,
    n_splits=5,
    join_strategy="left",
    aggregation="none",
    cuda=False,
):
    source_table, num_features, cat_features = em.prepare_table_for_evaluation(
        source_table, num_features
    )

    results_dict = {}
    if join_strategy == "nojoin":
        em.run_on_table_cross_valid(
            source_table,
            num_features,
            cat_features,
            scenario_logger,
            n_splits=n_splits,
            verbose=verbose,
            iterations=iterations,
            cuda=cuda,
        )

    else:
        # Run on all candidates
        em.execute_on_candidates(
            join_candidates,
            source_table,
            scenario_logger,
            num_features,
            cat_features,
            verbose=verbose,
            iterations=iterations,
            n_splits=n_splits,
            join_strategy=join_strategy,
            aggregation=aggregation,
            cuda=cuda,
        )

    # em.execute_full_join(
    #     join_candidates,
    #     source_table,
    #     source_metadata,
    #     num_features,
    #     verbose=verbose,
    #     iterations=iterations,
    # )

    return
