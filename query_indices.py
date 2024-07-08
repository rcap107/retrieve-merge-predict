'''
This script is used to query the retrieval methods. Since the querying step may take a long time and can be done offline,
the step is done separately from the execution of the pipeline. 

The script runs by reading a configuration file that is provided by the user. Sample query files for the different data
lakes and settings are provided in folder `config/retrieval/query`. 

Run the script as follows:
```
python query_indices.py path/to/config.toml
```

'''

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
    """Parse the configuration file from CLI. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", action="store", type=argparse.FileType("r"))

    return parser.parse_args()


def prepare_dirtree():
    """Prepare the dirtree to ensure that the required folders are found. 
    """
    os.makedirs("data/metadata/queries", exist_ok=True)
    os.makedirs("results/query_results", exist_ok=True)


if __name__ == "__main__":

    # Read the config file
    args = parse_args()
    prepare_dirtree()
    config = toml.load(args.config_file)

    # Get the parameters
    jd_methods = config["join_discovery_method"]
    data_lake_version = config["data_lake"]
    rerank = config.get("hybrid", False)
    # Iterations are used only to re-run the same configuration multiple times to log multiple runs and reduce variance
    # in the results
    iterations = config.get("iterations", 1)
    query_cases = config["query_cases"]

    # Load the metadata index. 
    mdata_index = get_metadata_index(data_lake_version)

    for it in tqdm(range(iterations), position=1):
        if "minhash" in jd_methods:
            # If the index is minhash, the index is loaded only once, then queried multiple times. 
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
                    # Exact matching must be loaded once per querying result. 
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
                    # Indexing was done by starmie, so here we are just loading the query results and converting 
                    # them  to the format used for the pipeline. 
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
                
                # The index has been loaded, now it is queried and the results are saved. 
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
