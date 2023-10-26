import argparse
import logging
import os
import pickle
from pathlib import Path

import git
import polars as pl

import src.methods.evaluation as em
import src.pipeline as pipeline
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import QUERY_RESULTS_PATH, MetadataIndex, RawDataset
from src.utils.logging import setup_run_logging

repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha


def prepare_logger():
    logger = logging.getLogger("main")
    logger_scn = logging.getLogger("scn_logger")

    if not logger_scn.hasHandlers():
        fh = logging.FileHandler("results/logs/main_log.log")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(message)s")
        fh.setFormatter(fh_formatter)

        logger_scn.addHandler(fh)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler_formatter = logging.Formatter("'%(asctime)s %(message)s'")
        console_handler.setFormatter(console_handler_formatter)

        logger.addHandler(console_handler)

    return logger, logger_scn


def parse_arguments(default=None):
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
        "--selected_indices", action="store", default="minhash", nargs="*", type=str
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
        "--feature_selection",
        action="store_true",
        help="If true, perform feature selection.",
    )

    parser.add_argument(
        "--model_selection",
        action="store_true",
        help="If true, perform model selection.",
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


def single_run(args, run_name=None):
    # TODO: clean this up
    join_parameters = {
        k: v for k, v in (vars(args)).items() if k in ["join_strategy", "aggregation"]
    }
    model_parameters = {
        k: v
        for k, v in (vars(args)).items()
        if k in ["od_type", "od_wait", "iterations"]
    }

    pipeline.prepare_dirtree()
    logger, logger_scn = prepare_logger()
    logger.info("Starting run.")

    data_lake_version = args.yadl_version
    metadata_dir = Path(f"data/metadata/{data_lake_version}")
    metadata_index_path = Path(
        f"data/metadata/_mdi/md_index_{data_lake_version}.pickle"
    )
    index_dir = Path(f"data/metadata/_indices/{data_lake_version}")

    query_tab_path = Path(args.source_table_path)
    if not query_tab_path.exists():
        raise FileNotFoundError(f"File {query_tab_path} not found.")

    tab_name = query_tab_path.stem

    query_result_path = "{}__{}__{}__{}.pickle".format(
        args.yadl_version,
        args.index_name,
        tab_name,
        args.query_column,
    )
    with open(Path(QUERY_RESULTS_PATH, query_result_path), "rb") as fp:
        query_result = pickle.load(fp)

    scl = ScenarioLogger(
        base_table=tab_name,
        git_hash=repo_sha,
        iterations=args.iterations,
        aggregation=args.aggregation,
        chosen_model=args.chosen_model,
        target_dl=args.yadl_version,
        n_splits=args.n_splits,
        top_k=args.top_k,
        exp_name=run_name,
        debug=args.debug,
    )

    # if not metadata_index_path.exists():
    #     raise FileNotFoundError(
    #         f"Path to metadata index {metadata_index_path} is invalid."
    #     )
    # mdata_index = MetadataIndex(data_lake_variant=data_lake_version, index_path=metadata_index_path)
    # scl.pretty_print()

    # # Query index
    # # Removing duplicate rows
    # scl.add_timestamp("start_querying")
    df = pl.read_parquet(query_tab_path).unique()
    # query_tab_metadata = RawDataset(
    #     query_tab_path.resolve(), "queries", "data/metadata/queries"
    # )
    # query_tab_metadata.save_metadata_to_json()

    # query_column = args.query_column
    # if query_column not in df.columns:
    #     raise pl.ColumnNotFoundError()

    # # if args.sample_size is not None and args.sample_size > 0:
    # #     query = df[query_column].sample(int(args.sample_size)).drop_nulls()
    # # else:
    # query = df[query_column].drop_nulls()

    # logger.info("Start querying")
    # query_results, candidates_by_index = pipeline.querying(
    #     query_tab_metadata.metadata,
    #     query_column,
    #     query,
    #     indices,
    #     mdata_index,
    #     args.top_k,
    # )
    # logger.info("End querying")
    # scl.add_timestamp("end_querying")

    # query_result_path = Path("results/generated_candidates")
    # os.makedirs(query_result_path, exist_ok=True)
    # with open(Path(query_result_path, f"{tab_name}.pickle"), "wb") as fp:
    #     pickle.dump(candidates_by_index, fp)

    if not args.dry_run:
        scl.add_timestamp("start_evaluation")
        logger.info("Starting evaluation.")

        em.evaluate_joins(
            scl,
            df,
            # join_candidates=candidates_by_index["minhash"],
            join_candidates=query_result.candidates,
            target_column="target",
            chosen_model=args.chosen_model,
            join_estimators=[
                # "no_join",
                # "highest_containment",
                # "best_single_join",
                # "full_join",
                "stepwise_greedy_join",
            ],
            join_parameters=join_parameters,
            model_parameters=model_parameters,
            n_splits=args.n_splits,
            greedy_iterations=args.greedy_iterations,
            epsilon=args.epsilon,
        )
        logger.info("End evaluation.")
        scl.add_timestamp("end_evaluation")
        scl.set_status("SUCCESS")
    scl.add_timestamp("end_process")
    scl.add_process_time()

    scl.finish_run()
    logger_scn.debug(scl.to_string())
    logger.info("Run end.")


if __name__ == "__main__":
    args = parse_arguments()
    run_name = setup_run_logging()
    single_run(args, run_name)
