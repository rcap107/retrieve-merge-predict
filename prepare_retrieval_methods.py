import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import toml

from src.utils.indexing import prepare_join_discovery_methods


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", action="store")
    parser.add_argument("--repeats", action="store", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_file_path = args.config_file
    if Path(config_file_path).exists():
        config = toml.load(config_file_path)
        for _ in range(args.repeats):
            prepare_join_discovery_methods(config)
    else:
        raise FileNotFoundError
