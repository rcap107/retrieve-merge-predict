import argparse
import pickle
from pathlib import Path

import polars as pl
from tqdm import tqdm

import src.utils.pipeline_utils as utils
from src.utils.data_structures import RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins
from src.utils.data_structures import ScenarioLogger

import logging

import git 

repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha


# prepare scenario logger
scenario_logger = logging.getLogger("scenario_logger")
scenario_logger.setLevel(logging.DEBUG)

log_format = "%(message)s"
res_formatter = logging.Formatter(fmt=log_format)

rfh = logging.FileHandler(filename="results/scenario_logger.log")
rfh.setFormatter(res_formatter)

scenario_logger.addHandler(rfh)



# preparing generic logger
log_format = "%(asctime)s - %(message)s"
logger = logging.getLogger("main_pipeline")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt=log_format)
fh = logging.FileHandler(filename="results/logging_runs.log")
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "yadl_version",
        action="store",
        type=str,
        # choices=["full", "binary", "seltab", "wordnet"],
    )

    parser.add_argument(
        "--source_table_path",
        action="store",
        type=str,
        required=True,
        help="Path to the source table.",
    )

    parser.add_argument(
        "--query_column",
        action="store",
        type=str,
        required=True,
        help="Column to be used when querying.",
    )

    parser.add_argument(
        "--sample_size",
        action="store",
        type=int,
        default=None,
        help="If provided, number of samples to be taken from the query column. ",
    )

    parser.add_argument(
        "--sampling_seed",
        action="store",
        type=int,
        default=42,
        help="Random seed to be used when sampling, for reproducbility.",
    )

    parser.add_argument(
        "--query_result_path",
        action="store",
        type=str,
        default=None,
        help="Path to use when saving the query results in pickle form. ",
    )

    parser.add_argument(
        "--iterations",
        action="store",
        type=int,
        default=1000,
        help="Number of iterations to be executed in the evaluation step.",
    )

    parser.add_argument(
        "--join_strategy",
        action="store",
        type=str,
        default="left",
        choices=["left", "right", "inner", "outer"],
        help="Number of iterations to be executed in the evaluation step.",
    )

    parser.add_argument(
        "--aggregation",
        action="store",
        type=str,
        default="none",
        choices=["none", "dedup", "dfs"],
        help="Number of iterations to be executed in the evaluation step.",
    )

    
    parser.add_argument(
        "--k_fold",
        action="store",
        type=int,
        default=5,
        help="Number of crossvalidation folds.",
    )




    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info("Run start.")
    
    args = parse_arguments()
    case = args.yadl_version
    logger.info(f"Working with version `{case}`")
    metadata_dir = Path(f"data/metadata/{case}")
    metadata_index_path = Path(f"data/metadata/_mdi/md_index_{case}.pickle")
    index_dir = Path(f"data/metadata/_indices/{case}")

    query_data_path = Path(args.source_table_path)
    if not query_data_path.exists():
        raise FileNotFoundError(f"File {query_data_path} not found.")

    tab_name = query_data_path.stem
    
    
    scl = ScenarioLogger(
        source_table = tab_name,
        git_hash=repo_sha,
        iterations= args.iterations,
        join_strategy=args.join_strategy,
        aggregation=args.aggregation,
        target_dl = args.yadl_version,
        k_fold = args.k_fold
    )
    
    logger.info(f"Reading metadata from {metadata_index_path}")
    if not metadata_index_path.exists():
        raise FileNotFoundError(
            f"Path to metadata index {metadata_index_path} is invalid."
        )
    else:
        mdata_index = MetadataIndex(index_path=metadata_index_path)

    # print("Loading indices.")
    indices = utils.load_indices(index_dir)

    # Query index
    # print("Querying.")

    df = pl.read_parquet(query_data_path)
    logger.info(f"Querying from dataset {query_data_path}")
    query_metadata = RawDataset(
        query_data_path.resolve(), "queries", "data/metadata/queries"
    )
    query_metadata.save_metadata_to_json()

    query_column = args.query_column
    if query_column not in df.columns:
        raise pl.ColumnNotFoundError()

    if args.sample_size is not None:
        query = df[query_column].sample(int(args.sample_size)).drop_nulls()
    else:
        query = df[query_column].drop_nulls()

    logger.info("Querying start")
    query_results, candidates_by_index = utils.querying(
        query_metadata, query_column, query, indices, mdata_index
    )
    logger.info("Querying end")

    if args.query_result_path is not None:
        with open(args.query_result_path, "wb") as fp:
            pickle.dump(candidates_by_index, fp)
    else:
        with open(f"generated_candidates_{tab_name}.pickle", "wb") as fp:
            pickle.dump(candidates_by_index, fp)

    # TODO: Dropping profiling for a bit
    # print("Profiling results.")
    # profiling_results = profile_joins(candidates_by_index, logger=logger)

    logger.info("Evaluating join results.")
    results = utils.evaluate_joins(
        df,
        query_metadata,
        scl,
        join_candidates=candidates_by_index,
        num_features=None,
        verbose=0,
        iterations=args.iterations,
        join_strategy=args.join_strategy,
        aggregation=args.aggregation,
    )
    logger.info("Evaluation complete.")

    results["target_dl"] = args.yadl_version
    results_path = Path("results/run_results.csv")
    results.to_csv(
        results_path, mode="a", index=False, header=not results_path.exists()
    )
    scl.add_timestamp("end")
    scl.write_to_file("results/scenario_results.txt")
    scenario_logger.info(scl.to_string())
    logger.info("Run end.")
