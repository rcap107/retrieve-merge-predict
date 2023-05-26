from src.utils.data_structures import RawDataset
from src.data_preparation.utils import MetadataIndex
import src.utils.pipeline_utils as utils
import pandas as pd
import polars as pl
import os
from pathlib import Path

import logging

log_format = "%(asctime)s - %(message)s"

logger = logging.getLogger(__name__)
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


# ds_name = "us-accidents"
# data_folder = Path("data/source_tables/ken_datasets")
# src_dataset_path = Path(data_folder, ds_name)
# dataset_path  = Path(src_dataset_path, ds_name + ".csv")

# save_single_table(dataset_path, "ken_datasets", "data/metadata/sources")

# ds_name = "movies-prepared"
# data_folder = Path("data/source_tables/ken_datasets")
# src_dataset_path = Path(data_folder, "the-movies-dataset")
# dataset_path  = Path(src_dataset_path, ds_name + ".parquet")

# save_single_table(dataset_path, "ken_datasets", "data/metadata/sources")

logger.info("Starting metadata creation.")
for case in ["wordnet", "binary", "seltab"]:
    logger.debug(f"Case {case}")
    data_folder = Path(f"data/yago3-dl/{case}")
    if data_folder.exists():
        for dataset_path in data_folder.glob("**/*.parquet"):
            ds = RawDataset(dataset_path, "yago3-dl", f"data/metadata/{case}")
            ds.save_metadata_to_json(f"data/metadata/{case}")
            ds.save_metadata_to_json("data/metadata/full")
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")
logger.debug("Done")

# Prepare metadata indices
logger.info("Preparing metadata indices.")
for case in ["wordnet", "binary", "seltab", "full"]:
    metadata_index = MetadataIndex(f"data/metadata/{case}")
    metadata_index.save_index(f"data/metadata/_mdi/md_index_{case}.pickle")


# Prepare indices
index_dir = Path("data/metadata/_indices/")

selected_indices = ["minhash"]

logger.info("Preparing minhash indices")
for case in ["binary"]:
# for case in ["wordnet", "binary", "seltab", "full"]:
    logger.debug(f"Case: {case}")
    metadata_dir = Path(f"data/metadata/{case}")
    case_dir = Path(index_dir, case)
    os.makedirs(case_dir, exist_ok=True)

    index_configurations = utils.prepare_default_configs(metadata_dir, selected_indices)
    print("Preparing indices.")
    indices = utils.prepare_indices(index_configurations)
    print("Saving indices.")
    utils.save_indices(indices, index_dir)
logger.debug("Done")
