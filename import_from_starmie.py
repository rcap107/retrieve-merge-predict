# %%
import argparse
from pathlib import Path

from src.data_structures.retrieval_methods import StarmieWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_path", type=str, help="Path to STARMIE dir containing the query results."
    )
    parser.add_argument(
        "data_lake_version", type=str, help="Data lake version to evaluate."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_lake_version = args.data_lake_version
    root_path = args.base_path

    base_path = Path(root_path, data_lake_version)

    for pth in base_path.glob("*.parquet"):
        print(pth)
        base_table_path = Path(pth.stem.replace("query-results_cl_", "")).with_suffix(
            ".parquet"
        )
        output_dir = Path(f"data/metadata/_indices/{data_lake_version}")
        wr = StarmieWrapper(import_path=pth, base_table_path=base_table_path)
        wr.save_index(output_dir)
