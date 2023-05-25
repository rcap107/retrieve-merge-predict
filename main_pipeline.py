# %%
# %load_ext autoreload
# %autoreload 2
import argparse
import pickle
from pathlib import Path

import polars as pl
from tqdm import tqdm

import src.utils as utils
from src.data_preparation.data_structures import RawDataset
from src.data_preparation.utils import MetadataIndex
from src.table_integration.join_profiling import profile_joins


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
        "--sampling_seed",
        action="store",
        type=int,
        default=42,
        help="Random seed to be used when sampling, for reproducbility.",
    )

    args = parser.parse_args()
    return args


# %%
if __name__ == "__main__":
    args = parse_arguments()

    metadata_dir = "data/metadata/binary"
    mdata_index_path = Path("data/metadata/mdi/md_index_binary.pickle")

    index_dir = "data/metadata/indices"

    precomputed_indices = True  # if true, load indices from disk

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
    query_data_path = Path("data/source_tables/ken_datasets/presidential-results")
    tab_name = "presidential-results-prepared"
    tab_path = Path(query_data_path, f"{tab_name}.parquet")
    df = pl.read_parquet(tab_path)

    print(f"Querying from dataset {tab_path}")
    query_metadata = RawDataset(tab_path.resolve(), "queries", "data/metadata/queries")
    query_metadata.save_metadata_to_json()

    query_column = "col_to_embed"
    # query = df[query_column].sample(3000).drop_nulls()
    query = df[query_column].drop_nulls()

    query_results, candidates_by_index = utils.querying(
        query_metadata, query_column, query, indices, mdata_index
    )

    with open(f"generated_candidates_{tab_name}.pickle", "wb") as fp:
        pickle.dump(candidates_by_index, fp)

    print("Profiling results.")
    profiling_results = profile_joins(candidates_by_index)
    profiling_results.to_csv("results/profiling_results.csv", index=False, mode="a")

    print("Evaluating join results.")
    # TODO: Somewhere in here there still are issues with types
    results = utils.evaluate_joins(
        df, join_candidates=candidates_by_index, num_features=None, verbose=0
    )
    results.to_csv("results/run_results.csv", mode="a")
