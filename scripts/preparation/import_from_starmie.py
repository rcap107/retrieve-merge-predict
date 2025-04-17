"""
Simple script used to import the results obtained by Starmie.

Given the path to the Starmie results, this script builds the data structures
needed to run the pipeline.
"""

import argparse
from pathlib import Path

from src.data_structures.retrieval_methods import StarmieWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_path", type=Path, help="Path to STARMIE dir containing the query results."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    root_path = args.base_path
    data_lake_version = root_path.stem

    print(root_path)

    for pth in root_path.glob("*.parquet"):
        print(pth)
        base_table_path = Path(pth.stem.replace("starmie-cl_", "")).with_suffix(
            ".parquet"
        )
        output_dir = Path(f"data/metadata/_indices/{data_lake_version}")
        wr = StarmieWrapper(import_path=pth, base_table_path=base_table_path)
        wr.save_index(output_dir)
