import datetime as dt
import os
from pathlib import Path

import polars as pl
from memory_profiler import memory_usage, profile
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
    time_save = 0
    time_create = 0
    for query_case in tqdm(queries, position=1, total=len(queries)):
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


def test_retrieval_method(data_lake_version, index_name, queries, index_config):
    index_dir = Path(f"data/metadata/_indices/profiling/{data_lake_version}")
    os.makedirs(Path(index_dir), exist_ok=True)

    index_logger = SimpleIndexLogger(
        index_name=index_name,
        step="create",
        data_lake_version=data_lake_version,
        index_parameters=index_config,
    )

    rerank = index_config.pop("rerank", False)

    if index_name == "minhash":
        index_logger.start_time("create")
        mem_usage, this_index = memory_usage(
            (
                MinHashIndex,
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
                [queries, index_path, index_name, data_lake_version, rerank],
                {},
            ),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_load"] = time_load
        index_logger.durations["time_query"] = time_query

    elif index_name == "inverted_index":
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
                [queries, index_path, index_name, data_lake_version, rerank],
                {},
            ),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_load"] = time_load
        index_logger.durations["time_query"] = time_query

    elif index_name == "exact_matching":
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
    index_logger.write_to_json("results/profiling/retrieval")


if __name__ == "__main__":

    os.makedirs("data/metadata/_indices/profiling", exist_ok=True)
    os.makedirs("results/profiling/retrieval", exist_ok=True)

    data_lake_version = "wordnet_full"

    base_table_root = "data/source_tables/yadl/"

    retrieval_method = "minhash"

    queries = [
        (
            Path(base_table_root, "company_employees-yadl-depleted.parquet"),
            "col_to_embed",
        ),
        (Path(base_table_root, "housing_prices-yadl-depleted.parquet"), "col_to_embed"),
        (Path(base_table_root, "us_elections-yadl-depleted.parquet"), "col_to_embed"),
        (Path(base_table_root, "movies-yadl-depleted.parquet"), "col_to_embed"),
        (Path(base_table_root, "movies_vote-yadl-depleted.parquet"), "col_to_embed"),
        (Path(base_table_root, "us_accidents-yadl-depleted.parquet"), "col_to_embed"),
    ]

    if retrieval_method == "exact_matching":
        method_config = {
            "metadata_dir": [f"data/metadata/{data_lake_version}"],
            "n_jobs": [16],
        }
        cases = ParameterGrid(method_config)
        for config in cases:
            test_retrieval_method(data_lake_version, retrieval_method, queries, config)
    elif retrieval_method == "minhash":
        method_config = {
            "metadata_dir": [f"data/metadata/{data_lake_version}"],
            "n_jobs": [16],
            # "thresholds": [60],
            "thresholds": [20, 60, 80],
            "no_tag": [False],
            "rerank": [True],
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
            test_retrieval_method(data_lake_version, "inverted_index", queries, config)
