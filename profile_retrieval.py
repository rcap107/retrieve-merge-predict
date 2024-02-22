import datetime as dt
import os
from pathlib import Path

import polars as pl
from memory_profiler import memory_usage

from src.data_structures.loggers import SimpleIndexLogger
from src.data_structures.metadata import QueryResult, RawDataset
from src.data_structures.retrieval_methods import (
    ExactMatchingIndex,
    MinHashIndex,
    ReverseIndex,
)
from src.utils.indexing import get_metadata_index, load_index, query_index


def wrapper_prepare_exact_matching(queries, method_config, index_dir):
    time_save = 0
    time_create = 0
    for query_case in queries:
        method_config.update("base_table_path", query_case[0])
        method_config.update("query_column", query_case[1])
        start = dt.datetime.now()
        this_index = ExactMatchingIndex(**method_config)
        end = dt.datetime.now()
        time_create += (end - start).total_seconds()
        start = dt.datetime.now()
        this_index.save_index(index_dir)
        end = dt.datetime.now()
        time_save += (end - start).total_seconds()

    return time_create, time_save


def wrapper_query_index(queries, index_path, index_name):
    time_load = 0
    time_query = 0

    start = dt.datetime.now()
    if index_name == "minhash":
        this_index = MinHashIndex(index_file=index_path)
    elif index_name == "reverse_index":
        this_index = ReverseIndex(file_path=index_path)
    end = dt.datetime.now()
    time_load += (end - start).total_seconds()

    for query_case in queries:
        query_tab_path = Path(query_case[0])
        query_column = query_case[1]
        query_table = pl.read_parquet(query_tab_path)
        query = query_table[query_column]
        start = dt.datetime.now()
        this_index.query_index(query)
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
            data_lake_version,
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
            (wrapper_query_index, [queries, index_path, index_name], {}),
            timestamps=True,
            max_iterations=1,
            retval=True,
        )
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_load"] = time_load
        index_logger.durations["time_query"] = time_query

    elif index_name == "reverse_index":
        index_logger.start_time("create")
        mem_usage, this_index = memory_usage(
            (
                ReverseIndex,
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
            (wrapper_query_index, [queries, index_path, index_name], {}),
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
        index_logger.mark_memory(mem_usage, label="query")
        index_logger.durations["time_create"] = time_create
        index_logger.durations["time_save"] = time_save

    index_logger.to_logfile()
    index_logger.write_to_json("results/profiling/retrieval")


if __name__ == "__main__":

    os.makedirs("data/metadata/_indices/profiling", exist_ok=True)
    os.makedirs("results/profiling/retrieval", exist_ok=True)

    data_lake_version = "binary_update"

    queries = [
        (
            "data/source_tables/yadl/company_employees-yadl-depleted.parquet",
            "col_to_embed",
        ),
        (
            "data/source_tables/yadl/housing_prices-yadl-depleted.parquet",
            "col_to_embed",
        ),
        ("data/source_tables/yadl/us_elections-yadl-depleted.parquet", "col_to_embed"),
        ("data/source_tables/yadl/movies-yadl-depleted.parquet", "col_to_embed"),
        ("data/source_tables/yadl/movies_vote-yadl-depleted.parquet", "col_to_embed"),
        ("data/source_tables/yadl/us_accidents-yadl-depleted.parquet", "col_to_embed"),
    ]

    ## Index creation

    # # Minhash
    # config_minhash = {
    #     "metadata_dir": "data/metadata/binary_update",
    #     "n_jobs": 2,
    # }
    # prepare_index("minhash", config_minhash, data_lake_version)

    # # Exact matching
    # config_exact = {
    #     "metadata_dir": "data/metadata/binary_update",
    #     "base_table_path": "data/source_tables/yadl/us_elections_dems-yadl-depleted.parquet",
    #     "query_column": "col_to_embed",
    #     "n_jobs": 1,
    # }

    # prepare_index("exact_matching", config_exact, data_lake_version)

    # Reverse index
    method_config = {
        "metadata_dir": "data/metadata/wordnet_small",
        "n_jobs": 2,
        # "thresholds": 20,
    }

    test_retrieval_method(data_lake_version, "reverse_index", queries, method_config)
