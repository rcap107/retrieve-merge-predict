import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from joblib import Parallel, delayed
from tqdm import tqdm

import src.pipeline as pipeline
from src.data_structures.metadata import MetadataIndex, RawDataset
from src.utils.indexing import prepare_default_configs

log_format = "%(asctime)s - %(message)s"

logger = logging.getLogger("metadata_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt=log_format)

fh = logging.FileHandler(filename="results/logging_metadata.log")
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_indices",
        action="store_true",
        help="If true, save the indices after preparing the metadata.",
    )
    parser.add_argument(
        "case",
        action="store",
        metavar="CASE",
        help="Tag to be assigned to the current index.",
    )
    parser.add_argument(
        "data_folder",
        action="store",
        metavar="PATH",
        help="Path to the folder storing the tables to index.",
    )
    parser.add_argument(
        "--save_to_full",
        action="store_true",
        help="Save the current set of mdata to the full index. ",
    )

    parser.add_argument(
        "--selected_indices",
        action="store",
        choices=["manual", "minhash", "lazo"],
        nargs="*",
        default="minhash",
        help="Indices to prepare. ",
    )

    parser.add_argument(
        "--base_table",
        action="store",
        default=None,
    )

    parser.add_argument(
        "--n_jobs",
        action="store",
        default=1,
    )

    args = parser.parse_args()

    return args


def save_single_table(dataset_path, dataset_source, metadata_dest):
    ds = RawDataset(dataset_path, dataset_source, metadata_dest)
    ds.save_metadata_to_json()


def prepare_metadata_from_case(case, data_folder, save_to_full=False):
    logger.info("Case %s", case)
    data_folder = Path(data_folder)
    if data_folder.exists():
        os.makedirs(f"data/metadata/{case}", exist_ok=True)
        os.makedirs(f"data/metadata/_mdi", exist_ok=True)

        total_files = sum(1 for f in data_folder.glob("**/*.parquet"))

        metadata_dest = f"data/metadata/{case}"
        Parallel(n_jobs=-1, verbose=0)(
            delayed(save_single_table)(dataset_path, "yadl", metadata_dest)
            for dataset_path in tqdm(
                data_folder.glob("**/*.parquet"), total=total_files
            )
        )

        # for dataset_path in data_folder.glob("**/*.parquet"):
        #     ds = RawDataset(dataset_path, "yadl", )
        #     ds.save_metadata_to_json(f"data/metadata/{case}")
        metadata_index = MetadataIndex(data_lake_variant=case, f"data/metadata/{case}")
        metadata_index.save_index(f"data/metadata/_mdi/md_index_{case}.pickle")
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")


def prepare_indices(
    case,
    selected_indices=["minhash"],
    base_table_path=None,
    n_jobs=1,
):
    logger.info("Preparing indices")
    index_dir = Path(f"data/metadata/_indices/{case}")
    logger.info("Indices: start %s", case)
    metadata_dir = Path(f"data/metadata/{case}")
    case_dir = Path(index_dir, case)
    os.makedirs(case_dir, exist_ok=True)

    index_configurations = prepare_default_configs(metadata_dir, selected_indices)
    print("Preparing indices.")
    indices = pipeline.prepare_indices(index_configurations)
    print("Saving indices.")
    pipeline.save_indices(indices, index_dir)
    logger.info("Indices: end %s", case)


if __name__ == "__main__":
    # args = parse_args()
    a = {
        "case": "wordnet_big",
        "data_folder": "data/wordnet_big",
        "save_indices": True,
        "save_to_full": False,
        "selected_indices": ["minhash"],
        "base_table": None,
        "n_jobs": -1,
        "build_metadata": True,
    }

    args = SimpleNamespace(**a)
    if args.build_metadata:
        start_time = dt.datetime.now()
        logger.info("START - Metadata creation - %s" % args.case)
        prepare_metadata_from_case(args.case, args.data_folder, args.save_to_full)
        logger.info("END - Metadata creation - %s" % args.case)

    if args.save_indices:
        logger.info("START - Index creation")
        prepare_indices(args.case, args.selected_indices, args.base_table)
        logger.info("END - Preparing indices")
    end_time = dt.datetime.now()
    logger.info(
        f"SUMMARY - Time required (s): {(end_time - start_time).total_seconds():.2f}"
    )
