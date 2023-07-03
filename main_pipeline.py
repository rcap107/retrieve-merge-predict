import argparse
import logging
import os
import pickle
from pathlib import Path

import git
import polars as pl

import src.pipeline as utils
from src.data_structures.metadata import MetadataIndex, RawDataset
from src.data_structures.loggers import ScenarioLogger

repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha


def prepare_logger():
    logger = logging.getLogger("main")
    logger_scn = logging.getLogger("scn_logger")
    # file handler for scenario logs
    fh = logging.FileHandler("results/logs/main_log.log")
    fh.setLevel(logging.DEBUG)

    # console handler for info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # set formatter
    fh_formatter = logging.Formatter("%(message)s")
    fh.setFormatter(fh_formatter)

    ch_formatter = logging.Formatter("'%(asctime)s %(message)s'")
    ch.setFormatter(ch_formatter)

    # add handler to logger
    logger_scn.addHandler(fh)
    logger.addHandler(ch)

    return logger, logger_scn


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
        "--n_jobs",
        action="store",
        type=int,
        default=1,
        help="Number of parallel jobs to use during the training. `-1` to use all cores.",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run on GPU.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    utils.prepare_dirtree()

    logger, logger_scn = prepare_logger()
    logger.info("Starting run.")
    args = parse_arguments()
    case = args.yadl_version
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

    if not metadata_index_path.exists():
        raise FileNotFoundError(
            f"Path to metadata index {metadata_index_path} is invalid."
        )
    else:
        mdata_index = MetadataIndex(index_path=metadata_index_path)
    scl.add_timestamp("start_load_index")
    indices = utils.load_indices(index_dir)
    scl.add_timestamp("end_load_index")

    scl.pretty_print()

    # Query index
    # I am removing all duplicate rows
    scl.add_timestamp("start_querying")
    df = pl.read_parquet(query_data_path).unique()
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

    logger.info("Start querying")
    query_results, candidates_by_index = utils.querying(
        query_metadata, query_column, query, indices, mdata_index, args.top_k
    )
    logger.info("End querying")
    scl.add_timestamp("end_querying")

    scl.results["n_candidates"] = len(candidates_by_index["minhash"])

    if args.query_result_path is not None:
        with open(args.query_result_path, "wb") as fp:
            pickle.dump(candidates_by_index, fp)
    else:
        query_result_path = Path("results/generated_candidates")
        os.makedirs(query_result_path, exist_ok=True)
        with open(Path(query_result_path, f"{tab_name}.pickle"), "wb") as fp:
            pickle.dump(candidates_by_index, fp)

    if not args.dry_run:
        scl.add_timestamp("start_evaluation")
        logger.info("Starting evaluation.")
        utils.evaluate_joins(
            df,
            scl,
            join_candidates=candidates_by_index,
            verbose=0,
            iterations=args.iterations,
            n_splits=args.n_splits,
            join_strategy=args.join_strategy,
            aggregation=args.aggregation,
            cuda=args.cuda,
            n_jobs=args.n_jobs,
        )
        logger.info("End evaluation.")
        scl.add_timestamp("end_evaluation")

    # results["target_dl"] = args.yadl_version
    # results_path = Path("results/run_results.csv")
    # results.to_csv(
    #     results_path, mode="a", index=False, header=not results_path.exists()
    # )
    scl.add_timestamp("end_process")

    scl.write_to_file("results/scenario_results.txt")
    logger_scn.debug(scl.to_string())
    logger.info("Run end.")
