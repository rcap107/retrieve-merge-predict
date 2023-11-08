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
from src.utils.indexing import (
    prepare_default_configs,
    prepare_indices,
    save_indices,
    save_single_table,
)


def prepare_metadata_from_case(case, data_folder, save_to_full=False):
    # logger.info("Case %s", case)
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
        metadata_index = MetadataIndex(
            data_lake_variant=case, metadata_dir=f"data/metadata/{case}"
        )
        metadata_index.save_index(f"data/metadata/_mdi/md_index_{case}.pickle")
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")


if __name__ == "__main__":
    # args = parse_args()
    a = {
        "case": "wordnet_big",
        "data_folder": "data/wordnet_big",
        "save_to_full": False,
    }

    args = SimpleNamespace(**a)
    if args.build_metadata:
        start_time = dt.datetime.now()
        # logger.info("START - Metadata creation - %s" % args.case)
        prepare_metadata_from_case(args.case, args.data_folder, args.save_to_full)
        # logger.info("END - Metadata creation - %s" % args.case)
