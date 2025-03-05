"""
This script is used to profile the runtime and peak memory usage of the different 
retrieval methods. It incorporates various functions to improve the reliability 
of the measurements: it is possible to repeat the same operations multiple times 
to reduce the variance in the results.

To improve reliability, this script first builds the index and then queries it: 
it will overwrite any pre-built index in each iteration.

Note that Starmie is profiled in a different repository, so it is not included here.
"""

import argparse
import datetime as dt
import os
from pathlib import Path

from memory_profiler import memory_usage
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.data_structures.loggers import SimpleIndexLogger
from src.data_structures.metadata import QueryResult, RawDataset
from src.data_structures.retrieval_methods import (
    ExactMatchingIndex,
    InvertedIndex,
    MinHashIndex,
)
from src.utils.indexing import get_metadata_index


def wrapper_prepare_exact_matching(queries, method_config, index_dir):
    """Utility function to collect the functions required to prepare Exact Matching."""
    time_save = 0
    time_create = 0
    for query_case in tqdm(queries, position=1, total=len(queries), desc="Query: "):
        method_config.update({"base_table_path": query_case[0]})
        method_config.update({"query_column": query_case[1]})

        start = dt.datetime.now()
        this_index = ExactMatchingIndex(**method_config)
        end = dt.datetime.now()
        time_create += (end - start).total_seconds()
        start = dt.datetime.now()
        this_index.save_index(index_dir)
        end = dt.datetime.now()
        time_save += (end - start).total_seconds()

    return time_create, time_save


def wrapper_query_index(queries, index_path, index_name, data_lake_version, rerank):
    time_load = 0
    time_query = 0

    metadata_index = get_metadata_index(data_lake_version)

    start = dt.datetime.now()
    if index_name.startswith("minhash"):
        this_index = MinHashIndex(index_file=index_path)
    elif index_name == "inverted_index":
        this_index = InvertedIndex(file_path=index_path)
    end = dt.datetime.now()
    time_load += (end - start).total_seconds()

    for query_case in queries:
        query_tab_path = Path(query_case[0])
        query_column = query_case[1]
        start = dt.datetime.now()
        query_tab_metadata = RawDataset(
            query_tab_path.resolve(), "queries", "data/metadata/queries"
        )
        query_result = QueryResult(
            this_index,
            query_tab_metadata,
            query_column,
            metadata_index,
            rerank,
            top_k=-1,
        )
        query_result.save_to_pickle("results/profiling/query_results")
        end = dt.datetime.now()
        time_query += (end - start).total_seconds()

    return time_load, time_query


def wrapper_query_exact_matching(queries, index_dir, data_lake_version):
    time_load = 0
    time_query = 0
    for query_case in queries:
        query_tab_path = Path(query_case[0])
        tname = Path(query_tab_path).stem
        query_column = query_case[1]

        index_path = Path(
            index_dir,
            f"em_index_{tname}_{query_column}.pickle",
        )

        start = dt.datetime.now()
        this_index = ExactMatchingIndex(file_path=index_path)
        end = dt.datetime.now()
        time_load += (end - start).total_seconds()
        start = dt.datetime.now()
        this_index.query_index(query_column)
        end = dt.datetime.now()
        time_query += (end - start).total_seconds()

    return time_load, time_query


def test_retrieval_method(data_lake_version, retrieval_method, queries, index_config):
    index_dir = Path(f"data/metadata/_indices/profiling/{data_lake_version}")
    os.makedirs(Path(index_dir), exist_ok=True)

    rerank = index_config.pop("rerank", False)

    if retrieval_method == "minhash" and rerank:
        logger_name = "minhash_hybrid"
    else:
        logger_name = retrieval_method

    index_logger = SimpleIndexLogger(
        index_name=logger_name,
        step="create",
        data_lake_version=data_lake_version,
        index_parameters=index_config,
    )

    if retrieval_method == "minhash":
        index_logger.start_time("create")
        # mem_usage, this_index = memory_usage(
        #     (
        #         MinHashIndex,
        #         [],
        #         index_config,
        #     ),
        #     timestamps=True,
        #     max_iterations=1,
        #     retval=True,
        # )
        index_logger.end_time("create")
        # index_logger.mark_memory(mem_usage, label="create")
        index_logger.start_time("save")
        # index_path = this_index.save_index(index_dir)
        index_logger.end_time("save")

        index_path = Path(
            index_dir,
            logger_name + "_20" + ".pickle",
        )


        mem_usage, (time_load, time_query) = memory_usage(
            (
                wrapper_query_index,
                [queries, index_path, retrieval_method, data_lake_version, rerank],
                {},
            ),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_load"] = time_load
        index_logger.durations["time_query"] = time_query

    elif retrieval_method == "inverted_index":
        index_logger.start_time("create")
        mem_usage, this_index = memory_usage(
            (
                InvertedIndex,
                [],
                index_config,
            ),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.end_time("create")
        index_logger.mark_memory(mem_usage, label="create")
        index_logger.start_time("save")
        index_path = this_index.save_index(index_dir)
        index_logger.end_time("save")

        mem_usage, (time_load, time_query) = memory_usage(
            (
                wrapper_query_index,
                [queries, index_path, retrieval_method, data_lake_version, rerank],
                {},
            ),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_load"] = time_load
        index_logger.durations["time_query"] = time_query

    elif retrieval_method == "exact_matching":
        mem_usage, (time_create, time_save) = memory_usage(
            (wrapper_prepare_exact_matching, [queries, index_config, index_dir], {}),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="create")
        index_logger.durations["time_create"] = time_create
        index_logger.durations["time_save"] = time_save

        mem_usage, (time_load, time_query) = memory_usage(
            (
                wrapper_query_exact_matching,
                [queries, index_dir, data_lake_version],
                {},
            ),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_load"] = time_load
        index_logger.durations["time_query"] = time_query

    index_logger.to_logfile()
    index_logger.write_to_json(f"results/profiling/retrieval/{retrieval_method}/{data_lake_version}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_lake_version", action="store")
    parser.add_argument("--retrieval_method", action="store")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Set True with retrieval_method=minhash to test hybrid minhash.",
    )
    parser.add_argument("--n_iter", action="store", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    data_lake_version = args.data_lake_version
    retrieval_method = args.retrieval_method
    n_iter = args.n_iter

    os.makedirs("data/metadata/_indices/profiling", exist_ok=True)
    os.makedirs("results/profiling/retrieval", exist_ok=True)

    os.makedirs(f"results/profiling/retrieval/{retrieval_method}/{data_lake_version}", exist_ok=True)

    # Open Data US has specific queries, so they're prepared explicitly here.
    if data_lake_version == "open_data_us":
        base_table_root = "data/source_tables/open_data_us"
        queries = [
            (
                Path(
                    base_table_root, "company_employees-depleted_name-open_data.parquet"
                ),
                "name",
            ),
            (
                Path(
                    base_table_root, "housing_prices-depleted_County-open_data.parquet"
                ),
                "County",
            ),
            (
                Path(
                    base_table_root,
                    "us_elections-depleted_county_name-open_data.parquet",
                ),
                "county_name",
            ),
            (
                Path(
                    base_table_root,
                    "us_accidents_2021-depleted-open_data_County.parquet",
                ),
                "County",
            ),
            (
                Path(base_table_root, "schools-depleted-open_data.parquet"),
                "col_to_embed",
            ),
        ]

    else:
        # All YADL data lakes have the same format for the queries.
        base_table_root = "data/source_tables/yadl/"
        queries = [
            (
                Path(base_table_root, "company_employees-yadl-depleted.parquet"),
                "col_to_embed",
            ),
            (
                Path(base_table_root, "housing_prices-yadl-depleted.parquet"),
                "col_to_embed",
            ),
            (
                Path(base_table_root, "us_elections-yadl-depleted.parquet"),
                "col_to_embed",
            ),
            (
                Path(base_table_root, "us_accidents_2021-yadl-depleted.parquet"),
                "col_to_embed",
            ),
            (
                Path(base_table_root, "us_county_population-yadl-depleted.parquet"),
                "col_to_embed",
            ),
        ]

    for i in tqdm(range(n_iter), total=n_iter, desc="Iteration: ", position=2):
        # In the following: add new parameters to build a parameter grid and test all combinations.
        if retrieval_method == "exact_matching":
            method_config = {
                "metadata_dir": [f"data/metadata/{data_lake_version}"],
                "n_jobs": [1],
            }
            cases = ParameterGrid(method_config)
            for config in cases:
                test_retrieval_method(
                    data_lake_version, retrieval_method, queries, config
                )
        elif retrieval_method == "minhash":
            method_config = {
                "metadata_dir": [f"data/metadata/{data_lake_version}"],
                "n_jobs": [32],
                "thresholds": [20],
                "no_tag": [False],
                "rerank": [args.rerank],
            }
            cases = ParameterGrid(method_config)
            for config in cases:
                test_retrieval_method(data_lake_version, "minhash", queries, config)
        elif retrieval_method == "inverted_index":
            method_config = {
                "metadata_dir": [f"data/metadata/{data_lake_version}"],
                "n_jobs": [16],
            }
            cases = ParameterGrid(method_config)
            for config in cases:
                test_retrieval_method(
                    data_lake_version, "inverted_index", queries, config
                )
