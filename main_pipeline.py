import argparse
import pickle
from pathlib import Path

import polars as pl
from tqdm import tqdm

import src.utils.pipeline_utils as utils
from src.utils.data_structures import RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins
from src.utils.logging_utils import RunLogger


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata_dir",
        action="store",
        type=str,
        required=True,
        help="Directory that stores the metadata of the data lake variant to be used. ",
    )

    parser.add_argument(
        "--metadata_index",
        action="store",
        type=str,
        required=True,
        help="Path to the metadata index. ",
    )

    parser.add_argument(
        "--index_dir",
        action="store",
        type=str,
        default="data/metadata/indices",
        help="Directory storing the index information.",
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    metadata_dir = Path(args.metadata_dir)
    mdata_index_path = Path(args.metadata_index)

    index_dir = args.index_dir

    precomputed_indices = True  # if true, load indices from disk

    logger = RunLogger()

    print(f"Reading metadata from {mdata_index_path}")
    if not mdata_index_path.exists():
        raise FileNotFoundError(
            f"Path to metadata index {mdata_index_path} is invalid."
        )
    else:
        mdata_index = MetadataIndex(index_path=mdata_index_path)

    selected_indices = ["minhash", "lazo"]

    # TODO: move index construction to a different script
    if not precomputed_indices:
        index_configurations = utils.prepare_default_configs(
            metadata_dir, selected_indices
        )
        print("Preparing indices.")
        indices = utils.prepare_indices(index_configurations)
        print("Saving indices.")
        utils.save_indices(indices, index_dir)
    else:
        print("Loading indices.")
        indices = utils.load_indices(index_dir)

    # Query index
    print("Querying.")
    query_data_path = Path(args.source_table_path)
    if not query_data_path.exists():
        raise FileNotFoundError(f"File {query_data_path} not found.")

    tab_name = query_data_path.stem

    df = pl.read_parquet(query_data_path)
    print(f"Querying from dataset {query_data_path}")
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

    logger.add_time("start_querying")
    query_results, candidates_by_index = utils.querying(
        query_metadata, query_column, query, indices, mdata_index
    )
    logger.add_time("end_querying")

    if args.query_result_path is not None:
        with open(args.query_result_path, "wb") as fp:
            pickle.dump(candidates_by_index, fp)
    else:
        with open(f"generated_candidates_{tab_name}.pickle", "wb") as fp:
            pickle.dump(candidates_by_index, fp)

    # TODO: Dropping profiling for a bit
    # print("Profiling results.")
    # profiling_results = profile_joins(candidates_by_index, logger=logger)

    print("Evaluating join results.")
    # TODO: Somewhere in here there still are issues with types
    results = utils.evaluate_joins(
        df,
        query_metadata,
        join_candidates=candidates_by_index,
        num_features=None,
        verbose=0,
        iterations=args.iterations,
    )
    results.to_csv("results/run_results.csv", mode="a")

    logger.save_logger()
