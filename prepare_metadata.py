import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from joblib import Parallel, delayed
from tqdm import tqdm

from src.data_structures.metadata import MetadataIndex
from src.utils.indexing import save_single_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", action="store")

    args = parser.parse_args()
    return args


def prepare_metadata_from_case(data_folder):
    # logger.info("Case %s", case)
    data_folder = Path(data_folder)
    case = data_folder.stem
    if data_folder.exists():
        os.makedirs(f"data/metadata/{case}", exist_ok=True)
        os.makedirs("data/metadata/_mdi", exist_ok=True)

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
    args = parse_args()
    # a = {
    #     "case": "binary_update",
    #     "data_folder": "data/yadl/binary_update",
    # }

    # args = SimpleNamespace(**a)
    prepare_metadata_from_case(args.data_folder)
