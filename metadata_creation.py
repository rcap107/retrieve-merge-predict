"""
One-time script to be run on each data lake to prepare the metadata that is used for all successive operations. 

Provide a path to the root data folder that contains the data lake, then the script will recursively explore all folders
and add all files with  ".parquet" extension to the metadata index.

The metadata index is used throughout the pipeline to identify each table a data lake. 
"""

import argparse
import os
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_data_folder",
        action="store",
        help="Path to the folder to use to prepare metadata. ",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="If provided, do not proceed recursively and only consider parquet files found in the main folder.",
    )

    return parser.parse_args()


def prepare_metadata_from_case(data_folder, flat=False):
    data_folder = Path(data_folder)
    case = data_folder.stem
    if flat:
        case += "_flat"
    if data_folder.exists() and data_folder.is_dir():
        print(f"Building metadata for folder {data_folder}")
        # Importing here to save time, so that if the folder is missing we find out immediately.
        from src.data_structures.metadata import MetadataIndex
        from src.utils.indexing import save_single_table

        # Creating the dirtree
        os.makedirs(f"data/metadata/{case}", exist_ok=True)
        os.makedirs("data/metadata/_mdi", exist_ok=True)

        match_pattern = "**/*.parquet" if not flat else "*.parquet"
        total_files = sum(1 for f in data_folder.glob(match_pattern))

        if total_files == 0:
            raise RuntimeError("No parquet files found. Is the path correct? ")

        metadata_dest = f"data/metadata/{case}"
        
        # Prepare the metadata by running on all available cores
        Parallel(n_jobs=-1, verbose=0)(
            delayed(save_single_table)(dataset_path, "yadl", metadata_dest)
            for dataset_path in tqdm(data_folder.glob(match_pattern), total=total_files)
        )
        metadata_index = MetadataIndex(
            data_lake_variant=case, metadata_dir=f"data/metadata/{case}"
        )
        metadata_index.save_index(f"data/metadata/_mdi/md_index_{case}.pickle")
    else:
        raise FileNotFoundError(f"Invalid data folder path '{data_folder}'")


if __name__ == "__main__":
    args = parse_args()
    prepare_metadata_from_case(args.path_data_folder, args.flat)
