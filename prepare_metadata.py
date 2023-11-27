import argparse
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from joblib import Parallel, delayed
from tqdm import tqdm

from src.data_structures.metadata import MetadataIndex
from src.utils.indexing import save_single_table


def prepare_metadata_from_case(data_folder):
    # logger.info("Case %s", case)
    data_folder = Path(data_folder)
    if data_folder.exists():
        metadata_index = MetadataIndex(data_lake_path=data_folder, n_jobs=1)
        metadata_index.create_index()
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")


if __name__ == "__main__":
    a = {
        "case": "binary_update",
        "data_folder": "data/yadl/binary_update",
    }

    args = SimpleNamespace(**a)
    prepare_metadata_from_case(args.data_folder)
