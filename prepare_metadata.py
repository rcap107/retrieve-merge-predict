import logging
import os
from pathlib import Path

import pandas as pd
import polars as pl
from tqdm import tqdm

import src.utils.pipeline_utils as utils
from src.data_preparation.utils import MetadataIndex
from src.utils.data_structures import RawDataset

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


def save_single_table(dataset_path, dataset_source, metadata_dest):
    ds = RawDataset(dataset_path, dataset_source, metadata_dest)
    ds.save_metadata_to_json()


def prepare_yadl_versions(cases=[], save_to_full=False):
    for case in cases:
        logger.info("Case %s", case)
        data_folder = Path(f"data/yago3-dl/{case}")
        if data_folder.exists():
            os.makedirs(f"data/metadata/{case}", exist_ok=True)
            for dataset_path in data_folder.glob("**/*.parquet"):
                ds = RawDataset(dataset_path, "yago3-dl", f"data/metadata/{case}")
                ds.save_metadata_to_json(f"data/metadata/{case}")
                if save_to_full:
                    ds.save_metadata_to_json("data/metadata/full")
        else:
            raise FileNotFoundError(f"Invalid path {data_folder}")

    # for case in cases:
    #     metadata_index = MetadataIndex(f"data/metadata/{case}")
    #     metadata_index.save_index(f"data/metadata/_mdi/md_index_{case}.pickle")


def prepare_gittables():
    case = "gittables"
    logger.info("Case %s", case)
    data_folder = Path("data/gittables/extracted/")
    if data_folder.exists():
        total = sum(1 for _ in data_folder.glob("**/*.parquet"))
        for dataset_path in tqdm(data_folder.glob("**/*.parquet"), total=total):
            ds = RawDataset(dataset_path, "gittables", f"data/metadata/{case}")
            ds.save_metadata_to_json(f"data/metadata/{case}")
            # ds.save_metadata_to_json("data/metadata/full")
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")

def prepare_auctus():
    case = "auctus"
    logger.info("Case %s", case)
    data_folder = Path("data/auctus/parquet/")
    if data_folder.exists():
        total = sum(1 for _ in data_folder.glob("**/*.parquet"))
        for dataset_path in tqdm(data_folder.glob("**/*.parquet"), total=total):
            ds = RawDataset(dataset_path, "auctus", f"data/metadata/{case}")
            ds.save_metadata_to_json(f"data/metadata/{case}")
            # ds.save_metadata_to_json("data/metadata/full")
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")


def prepare_indices(cases=[]):
    
    for case in cases:
        metadata_index = MetadataIndex(f"data/metadata/{case}")
        metadata_index.save_index(f"data/metadata/_mdi/md_index_{case}.pickle")

    # Prepare indices
    selected_indices = ["minhash"]

    logger.info("Preparing minhash indices")
    # for case in ["wordnet"]:
    for case in cases:
        index_dir = Path(f"data/metadata/_indices/{case}")
        logger.info("Minhash: start %s", case)
        metadata_dir = Path(f"data/metadata/{case}")
        case_dir = Path(index_dir, case)
        os.makedirs(case_dir, exist_ok=True)

        index_configurations = utils.prepare_default_configs(
            metadata_dir, selected_indices
        )
        print("Preparing indices.")
        indices = utils.prepare_indices(index_configurations)
        print("Saving indices.")
        utils.save_indices(indices, index_dir)
        logger.info("Minhash: end %s", case)
    logger.info("Done")


logger.info("Starting metadata creation - YADL")
prepare_yadl_versions(cases=["wordnet_cp"], save_to_full=False)
logger.debug("Done with metadata - YADL")

# logger.info("Starting metadata creation - gittables")
# prepare_gittables()
# logger.debug("Done with metadata - gittables")

# logger.info("Starting metadata creation - auctus")
# prepare_auctus()
# logger.debug("Done with metadata - auctus")


# logger.info("Preparing indices")
prepare_indices(cases=["wordnet_cp"])
# logger.info("Done with indices")
