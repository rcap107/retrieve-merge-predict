import logging
import os
import pickle
from pathlib import Path

import polars as pl
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from src.data_structures.indices import LazoIndex, ManualIndex, MinHashIndex
from src.data_structures.metadata import CandidateJoin, MetadataIndex
from src.methods import evaluation as em

logger_sh = logging.getLogger("pipeline")


def prepare_dirtree():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/generated_candidates", exist_ok=True)
    os.makedirs("data/metadata/queries", exist_ok=True)


def prepare_default_configs(data_dir, selected_indices=None):
    """Prepare default configurations for various indexing methods and provide the
    data directory that contains the metadata of the tables to be indexed.

    Args:
        data_dir (str): Path to the directory that contains the metadata.
        selected_indices (str, optional): If provided, prepare and run only the selected indices.

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
                "thresholds": [20],
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


def prepare_config_manual(base_table_path, mdata_path, n_jobs):
    base_table = pl.read_parquet(base_table_path)
    return {
        "manual": {
            "df_base": base_table,
            "tab_name": Path(base_table_path).stem,
            "mdata_path": mdata_path,
            "n_jobs": n_jobs,
        }
    }


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
        elif index == "manual":
            index_dict[index] = ManualIndex(**config)
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
        elif iname == "manual":
            if "manual_" + tab_name in index_path.stem:
                index = ManualIndex()
                index.load_index(index_path=index_path)
        else:
            raise ValueError(f"Unknown index {iname}.")

    return index


def load_indices(index_dir, selected_indices=["minhash"], tab_name=None):
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

    if "manual" in selected_indices and tab_name is None:
        raise RuntimeError("'manual' index requires 'tab_name' to be non-null.")

    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise IOError(f"Index dir {index_dir} does not exist.")
    index_dict = {}

    for index_path in index_dir.glob("**/*.pickle"):
        with open(index_path, "rb") as fp:
            input_dict = pickle.load(fp)
            iname = input_dict["index_name"]
            if iname in selected_indices:
                if iname == "minhash":
                    index = MinHashIndex()
                    index.load_index(index_dict=input_dict)
                elif iname == "lazo":
                    index = LazoIndex()
                    index.load_index(index_path)
                elif iname == "manual":
                    if "manual_" + tab_name in index_path.stem:
                        index = ManualIndex()
                        index.load_index(index_path=index_path)
                    else:
                        continue
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
            source_table_metadata=mdata_source,
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


def evaluate_joins(
    base_table,
    scenario_logger,
    candidates_by_index: dict,
    verbose=1,
    iterations=1000,
    n_splits=5,
    test_size=0.25,
    aggregation="first",
    group_column="col_to_embed",
    split_kind="group_shuffle",
    top_k=5,
):
    # prepare_logger(scenario_logger.scenario_id)

    groups = base_table.select(
        pl.col(group_column).cast(pl.Categorical).cast(pl.Int16).alias("group")
    ).to_numpy()

    if split_kind == "group_shuffle":
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=None)
        splits = gss.split(base_table, groups=groups)
    elif split_kind == "shuffle":
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=None)
        splits = ss.split(base_table)
    else:
        raise ValueError(f"Inappropriate value {split_kind} for `split_kind`.")

    splits = list(splits)
    summary_results = []

    results = em.base_table(
        scenario_logger,
        splits,
        base_table,
        target_column="target",
        iterations=iterations,
        verbose=verbose,
    )
    summary_results.append(results)

    # Iterate over each index and run experiments on that
    for index_name, join_candidates in candidates_by_index.items():
        # Join on each candidate, one at a time
        results, df_ranking = em.single_join(
            scenario_logger,
            splits,
            join_candidates,
            index_name,
            base_table,
            iterations,
            aggregation=aggregation,
            verbose=verbose,
            top_k=top_k,
        )

        summary_results.append(results)

        # Join all candidates at the same time
        results = em.full_join(
            scenario_logger,
            splits,
            join_candidates,
            index_name,
            base_table,
            iterations=iterations,
            verbose=verbose,
            aggregation=aggregation,
        )

        summary_results.append(results)

    summary = pl.from_dicts(summary_results)
    print(f'SOURCE TABLE: {scenario_logger.get_parameters()["base_table"]}')
    print(summary)

    scenario_logger.set_results(summary)

    return
