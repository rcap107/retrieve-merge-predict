"""
This file contains all the "index" objects that are used during the retrieval step to find join candidates.

The objects defined here are used throughout the pipeline.
"""

import logging
import os
import pickle
from pathlib import Path

from joblib import load
from memory_profiler import memory_usage
from tqdm import tqdm

from src.data_structures.loggers import SimpleIndexLogger
from src.data_structures.metadata import (
    CandidateJoin,
    MetadataIndex,
    QueryResult,
    RawDataset,
)
from src.data_structures.retrieval_methods import (
    CountVectorizerIndex,
    ExactMatchingIndex,
    LazoIndex,
    MinHashIndex,
)

logger = logging.getLogger("join_discovery_logger")

DEFAULT_INDEX_DIR = Path("data/metadata/_indices")
DEFAULT_QUERY_RESULT_DIR = Path("results/query_results")


def save_single_table(dataset_path, dataset_source, metadata_dest):
    ds = RawDataset(dataset_path, dataset_source, metadata_dest)
    ds.save_metadata_to_json()


def write_candidates_on_file(candidates, output_file_path, separator=","):
    """This is a utility function that saves the list of candidates as csv in the given path.

    This can be used to export the candidates retrieved by an index so that they can be used in alternative methods.

    Args:
        candidates (dict): Dictionary that contains the candidates.
        output_file_path (str | Path): Path to use to save the file.
        separator (str, optional): Which separator to use. Defaults to ",".
    """
    with open(output_file_path, "w") as fp:
        fp.write("tbl_pth1,tbl1,col1,tbl_pth2,tbl2,col2\n")

        for cand in candidates.values():
            rstr = cand.get_joinpath_str(sep=separator)
            fp.write(rstr + "\n")


def get_metadata_index(data_lake_version: str) -> MetadataIndex:
    """Utility function that returns the MetadataIndex for a given data lake version.

    Args:
        data_lake_version (str): Which data lake version should be used.

    Raises:
        FileNotFoundError: Raised if the given path does not contain a valid metadata file.

    Returns:
        MetadataIndex: Index for the given data lake.
    """
    metadata_index_path = Path(
        f"data/metadata/_mdi/md_index_{data_lake_version}.pickle"
    )

    if not metadata_index_path.exists():
        raise FileNotFoundError(
            f"Path to metadata index {metadata_index_path} is invalid."
        )
    mdata_index = MetadataIndex(
        data_lake_variant=data_lake_version, index_path=metadata_index_path
    )

    return mdata_index


def generate_candidates(
    index_name: str,
    index_result: list,
    metadata_index: MetadataIndex,
    mdata_source: dict,
    query_column: str,
    top_k=15,
):
    """Utility function that returns the list of candidates from the given index and metadata file.

    Args:
        index_name (str): Name of the index.
        index_result (list): Query result provided by the user.
        metadata_index (MetadataIndex): Metadata index.
        mdata_source (dict): Metadata of the source table.
        query_column (str): Query column.
        top_k (int, optional): Number of candidates that should be retrieved.. Defaults to 15.

    Returns:
        dict: Dictionary that contains the candidates.
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
        ranking = [(k, v.similarity_score) for k, v in candidates.items()]
        clamped = [x[0] for x in sorted(ranking, key=lambda x: x[1], reverse=True)][
            :top_k
        ]

        candidates = {k: v for k, v in candidates.items() if k in clamped}
    return candidates


def prepare_retrieval_methods(index_configurations: dict):
    """Given a dict of index configurations, initialize the required indices.

    Args:
        index_configurations (dict): Dictionary that contains the required configurations.

    Raises:
        NotImplementedError: Raise NotImplementedError when providing an index that is not recognized.

    """

    for index, config in index_configurations.items():
        for i_conf in tqdm(config, total=len(config), position=1):
            metadata_dir = Path(i_conf["metadata_dir"])
            data_lake_version = metadata_dir.stem
            if "thresholds" in i_conf:
                data_lake_version += f"_{i_conf['thresholds']}"
            index_dir = Path(f"data/metadata/_indices/{data_lake_version}")
            os.makedirs(index_dir, exist_ok=True)
            index_logger = SimpleIndexLogger(
                index_name=index,
                step="create",
                data_lake_version=data_lake_version,
                index_parameters=i_conf,
            )

            if "base_table_path" in i_conf:
                tqdm.write(f"Table: {Path(i_conf['base_table_path']).stem}")
                logger.info(
                    "Index creation start: %s - %s - %s"
                    % (data_lake_version, index, i_conf["base_table_path"])
                )
            else:
                logger.info(
                    "Index creation start: %s - %s " % (data_lake_version, index)
                )

            index_logger.start_time("create")
            if index == "lazo":
                mem_usage, this_index = memory_usage(
                    (LazoIndex, [], i_conf),
                    retval=True,
                    timestamps=True,
                )
                index_logger.mark_memory(mem_usage, "create")
                # this_index = LazoIndex(**i_conf)
            elif index == "minhash":
                mem_usage, this_index = memory_usage(
                    (MinHashIndex, [], i_conf),
                    retval=True,
                    timestamps=True,
                )
                index_logger.mark_memory(mem_usage, "create")

                # this_index = MinHashIndex(**i_conf)
            elif index == "count_vectorizer":
                mem_usage, this_index = memory_usage(
                    (CountVectorizerIndex, [], i_conf),
                    retval=True,
                    timestamps=True,
                )
                index_logger.mark_memory(mem_usage, "create")

                # this_index = CountVectorizerIndex(**i_conf)
            elif index == "exact_matching":
                mem_usage, this_index = memory_usage(
                    (ExactMatchingIndex, [], i_conf),
                    retval=True,
                    timestamps=True,
                )
                index_logger.mark_memory(mem_usage, "create")

                # this_index = ExactMatchingIndex(**i_conf)
            else:
                raise NotImplementedError
            index_logger.end_time("create")
            logger.info("Index creation end: %s - %s " % (data_lake_version, index))

            index_logger.start_time("save")
            this_index.save_index(index_dir)
            index_logger.end_time("save")
            query_table = Path(i_conf.get("base_table_path", "")).stem
            query_column = i_conf.get("query_column", "")

            index_logger.update_query_parameters(query_table, query_column)
            index_logger.to_logfile()


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


def load_index(config):
    jd_method = config["join_discovery_method"]
    data_lake_version = config["data_lake"]

    index_path = Path(DEFAULT_INDEX_DIR, data_lake_version, f"{jd_method}_index.pickle")

    if jd_method == "minhash":
        with open(index_path, "rb") as fp:
            input_dict = load(fp)
        index = MinHashIndex()
        index.load_index(index_dict=input_dict)
    elif jd_method == "lazo":
        index = LazoIndex()
        index.load_index(index_path)
    else:
        raise ValueError(f"Unknown index {jd_method}.")
    return index


def query_index(
    index: MinHashIndex | LazoIndex,
    query_tab_path,
    query_column,
    mdata_index,
    rerank: bool = False,
    index_logger: SimpleIndexLogger | None = None,
):
    """Query a given index to produce a list of candidates. It may take an IndexLogger object to track runtime and
    memory usage.

    Args:
        index (MinHashIndex | LazoIndex | ExactMatchingIndex): Index object to query.
        query_tab_path (Path | str  ): Path to the base table.
        query_column (_type_): Column that should be used for querying.
        mdata_index (_type_): MetadataIndex object.
        rerank (bool, optional): Parameter that is forwarded to QueryResult. Defaults to False.
        index_logger (SimpleIndexLogger | None, optional): Logger object that is used to track various metric about the operation. Defaults to None.

    Returns:
        _type_: _description_
    """
    query_tab_metadata = RawDataset(
        query_tab_path.resolve(), "queries", "data/metadata/queries"
    )
    query_tab_metadata.save_metadata_to_json()

    if index_logger is not None:
        index_logger.start_time("query")
    query_result = QueryResult(
        index, query_tab_metadata, query_column, mdata_index, rerank
    )
    query_result.save_to_pickle()

    if index_logger is not None:
        index_logger.end_time("query")

    return query_result, index_logger


def load_query_result(
    data_lake_version: str,
    index_name: str,
    tab_name: str,
    query_column: str,
    top_k: int = 0,
    validate: bool = False,
):
    query_result_path = Path(
        "{}__{}__{}__{}.pickle".format(
            data_lake_version,
            index_name,
            tab_name,
            query_column,
        )
    )

    query_path = Path(DEFAULT_QUERY_RESULT_DIR, data_lake_version, query_result_path)
    if not query_path.exists():
        raise ValueError(f"Query {query_path} not found.")

    if not (isinstance(top_k, int) and top_k >= 0):
        raise ValueError(f"Value '{top_k}' is not valid for variable top_k")
    with open(
        Path(DEFAULT_QUERY_RESULT_DIR, data_lake_version, query_result_path),
        # Path(DEFAULT_QUERY_RESULT_DIR, query_result_path),
        "rb",
    ) as fp:
        query_result = pickle.load(fp)

    if len(query_result) < 1:
        raise ValueError(
            f"Found no candidates for query: {data_lake_version} - {index_name} - {query_column}."
        )
    if top_k > 0:
        query_result.select_top_k(top_k)
    return query_result
