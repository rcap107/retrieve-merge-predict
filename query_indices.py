import argparse
import os
from pathlib import Path

import toml
from tqdm import tqdm

from src.data_structures.loggers import SimpleIndexLogger
from src.data_structures.retrieval_methods import ExactMatchingIndex, StarmieWrapper
from src.utils.indexing import (
    DEFAULT_INDEX_DIR,
    get_metadata_index,
    load_index,
    query_index,
)

PREFIX = {
    "exact_matching": "em_index",
    "starmie": "starmie_index",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", action="store", type=argparse.FileType("r"))

    return parser.parse_args()


def prepare_dirtree():
    os.makedirs("data/metadata/queries", exist_ok=True)
    os.makedirs("results/query_results", exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    prepare_dirtree()
    config = toml.load(args.config_file)

    jd_methods = config["join_discovery_method"]
    data_lake_version = config["data_lake"]
    rerank = config.get("hybrid", False)
    iterations = config.get("iterations", 1)
    query_cases = config["query_cases"]

    mdata_index = get_metadata_index(data_lake_version)

    for it in tqdm(range(iterations), position=1):
        if "minhash" in jd_methods:
            index_name = "minhash_hybrid" if rerank else "minhash"
            # The SimpleIndexLogger is used to track the time required to query each index
            logger_minhash = SimpleIndexLogger(
                index_name=index_name,
                step="query",
                data_lake_version=data_lake_version,
                log_path="results/query_logging.txt",
            )
            logger_minhash.start_time("load")
            minhash_index = load_index(
                {"join_discovery_method": "minhash", "data_lake": data_lake_version}
            )
            logger_minhash.end_time("load")

        for query_case in tqdm(query_cases, total=len(query_cases), leave=False):
            tname = Path(query_case["table_path"]).stem
            query_tab_path = Path(query_case["table_path"])
            query_column = query_case["query_column"]
            for jd_method in jd_methods:
                if jd_method == "minhash":
                    # If the index is minhash, it is loaded only once.
                    index = minhash_index
                    index_logger = logger_minhash
                elif jd_method == "exact_matching":
                    # Otherwise, the index is loaded once for each query and the logger is created with it.
                    index_logger = SimpleIndexLogger(
                        index_name=jd_method,
                        step="query",
                        data_lake_version=data_lake_version,
                        log_path="results/query_logging.txt",
                    )
                    index_path = Path(
                        DEFAULT_INDEX_DIR,
                        data_lake_version,
                        f"{PREFIX[jd_method]}_{tname}_{query_column}.pickle",
                    )
                    index_logger.start_time("load")
                    if jd_method == "exact_matching":
                        index = ExactMatchingIndex(file_path=index_path)
                    else:
                        index = StarmieWrapper(file_path=index_path)
                    index_logger.end_time("load")

                elif jd_method == "starmie":
                    index_logger = SimpleIndexLogger(
                        index_name=jd_method,
                        step="query",
                        data_lake_version=data_lake_version,
                        log_path="results/query_logging.txt",
                    )
                    index_path = Path(
                        DEFAULT_INDEX_DIR,
                        data_lake_version,
                        f"{PREFIX[jd_method]}-{tname}.pickle",
                    )
                    index_logger.start_time("load")
                    index = StarmieWrapper(file_path=index_path)
                    index_logger.end_time("load")
                else:
                    raise ValueError(f"Unknown jd_method {jd_method}")
                index_logger.update_query_parameters(tname, query_column)
                query_result, index_logger = query_index(
                    index,
                    query_tab_path,
                    query_column,
                    mdata_index,
                    rerank=rerank,
                    index_logger=index_logger,
                )
                # If index logger is not None, save on file
                if index_logger:
                    index_logger.to_logfile()
