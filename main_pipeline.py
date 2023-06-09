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

import os
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
        help="Path to use when saving the results of the index queries in pickle form. ",
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
        choices=["left", "right", "inner", "outer", "nojoin"],
        help="Join strategy to be used.",
    )

    parser.add_argument(
        "--aggregation",
        action="store",
        type=str,
        default="first",
        choices=["first", "mean", "dfs"],
        help="Number of iterations to be executed in the evaluation step.",
    )

    parser.add_argument(
        "--n_splits",
        action="store",
        type=int,
        default=5,
        help="Number of crossvalidation folds.",
    )

    parser.add_argument(
        "--top_k",
        action="store",
        type=int,
        default=0,
        help="Number of candidates to keep. If 0, keep all candidates.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip evaluation. Used to generate the candidates.",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run on GPU.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info("Run start.")

    args = parse_arguments()
    logger.info(args)
    case = args.yadl_version
    # logger.info(f"Working with version `{case}`")
    metadata_dir = Path(f"data/metadata/{case}")
    metadata_index_path = Path(f"data/metadata/_mdi/md_index_{case}.pickle")
    index_dir = Path(f"data/metadata/_indices/{case}")

    query_data_path = Path(args.source_table_path)
    if not query_data_path.exists():
        raise FileNotFoundError(f"File {query_data_path} not found.")

    tab_name = query_data_path.stem

    scl = ScenarioLogger(
        source_table=tab_name,
        git_hash=repo_sha,
        iterations=args.iterations,
        join_strategy=args.join_strategy,
        aggregation=args.aggregation,
        target_dl=args.yadl_version,
        n_splits=args.n_splits,
    )

    # logger.info(f"Reading metadata from {metadata_index_path}")
    if not metadata_index_path.exists():
        raise FileNotFoundError(
            f"Path to metadata index {metadata_index_path} is invalid."
        )
    else:
        mdata_index = MetadataIndex(index_path=metadata_index_path)

    # print("Loading indices.")
    scl.add_timestamp("start_load_index")
    indices = utils.load_indices(index_dir)
    scl.add_timestamp("end_load_index")

    scl.pretty_print()

    # Query index
    # print("Querying.")

    # I am removing all duplicate rows
    scl.add_timestamp("start_querying")
    df = pl.read_parquet(query_data_path).unique()
    # TODO: Fix logging
    # logger.info(f"Querying from dataset {query_data_path}")
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

    # logger.info("Querying start")
    query_results, candidates_by_index = utils.querying(
        query_metadata, query_column, query, indices, mdata_index, args.top_k
    )
    # logger.info("Querying end")
    scl.add_timestamp("end_querying")

    scl.results["n_candidates"] = len(candidates_by_index["minhash"])

    if args.query_result_path is not None:
        
        with open(args.query_result_path, "wb") as fp:
            pickle.dump(candidates_by_index, fp)
    else:
        query_result_path = Path("results/generated_candidates")
        os.makedirs(query_result_path, exist_ok=True)
        with open(Path(query_result_path,f"{tab_name}.pickle"), "wb") as fp:
            pickle.dump(candidates_by_index, fp)

    # TODO: Dropping profiling for a bit
    # print("Profiling results.")
    # profiling_results = profile_joins(candidates_by_index, logger=logger)

    if not args.dry_run:
        scl.add_timestamp("start_evaluation")
        # logger.info("Evaluating join results.")
        utils.evaluate_joins(
            df,
            query_metadata,
            scl,
            join_candidates=candidates_by_index,
            verbose=0,
            iterations=args.iterations,
            n_splits=args.n_splits,
            join_strategy=args.join_strategy,
            aggregation=args.aggregation,
            cuda=args.cuda,
        )
        # logger.info("Evaluation complete.")
        scl.add_timestamp("end_evaluation")

    # results["target_dl"] = args.yadl_version
    # results_path = Path("results/run_results.csv")
    # results.to_csv(
    #     results_path, mode="a", index=False, header=not results_path.exists()
    # )
    scl.add_timestamp("end_process")

    scl.write_to_file("results/scenario_results.txt")
    scenario_logger.info(scl.to_string())
    logger.info("Run end.")
