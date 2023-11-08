import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import toml
from joblib import Parallel, delayed
from tqdm import tqdm

import src.pipeline as pipeline
from src.data_structures.metadata import MetadataIndex, RawDataset
from src.utils.indexing import (
    prepare_default_configs,
    prepare_join_discovery_methods,
    save_indices,
)

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


def parse_config(config_file_path):
    if Path(config_file_path).exists():
        config = toml.load(config_file_path)
        return config
    raise FileNotFoundError


if __name__ == "__main__":
    config_file_path = "config/join_discovery/prep_debug.toml"
    config = parse_config(config_file_path)

    args = SimpleNamespace(**config)
    case = args.data_lake_variant

    index_dir = Path(f"data/metadata/_indices/{case}")

    # logger.info("START - Index creation")
    print("Preparing indices.")
    indices = prepare_join_discovery_methods(args)
    print("Saving indices.")

    save_indices(indices, index_dir)
    # logger.info("Indices: end %s", case)
    # logger.info("END - Preparing indices")
    end_time = dt.datetime.now()
    # logger.info(
    #     f"SUMMARY - Time required (s): {(end_time - start_time).total_seconds():.2f}"
    # )
