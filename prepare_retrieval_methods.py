import argparse
from pathlib import Path

import toml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", action="store", help="Path to the configuration file to be used."
    )
    parser.add_argument(
        "--repeats",
        action="store",
        type=int,
        default=1,
        help="Utility parameter that repeats the index creation for the given config. to measure avg. time.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_file_path = args.config_file
    if Path(config_file_path).exists():
        config = toml.load(config_file_path)
        print(f"Reading configuration from file {config_file_path}")
        from src.utils.indexing import prepare_retrieval_methods

        for _ in range(args.repeats):
            prepare_retrieval_methods(config)
    else:
        raise FileNotFoundError
