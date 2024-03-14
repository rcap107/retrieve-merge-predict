import os

os.environ["POLARS_MAX_THREADS"] = "32"

import argparse
import pprint
from datetime import datetime as dt

import toml

from src.pipeline import prepare_config_dict, single_run
from src.utils.logging import archive_experiment, get_exp_name, setup_run_logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path",
        action="store",
        help="Path of the config file to be used.",
    )

    parser.add_argument(
        "-a",
        "--archive",
        required=False,
        action="store_true",
        help="If specified, archive the current run.",
    )

    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        help="If specified, skip writing logging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results/logs", exist_ok=True)

    start_run = dt.now()
    base_config = toml.load(args.input_path)
    run_variants = prepare_config_dict(base_config)

    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name(debug=args.debug)
    for idx, dd in enumerate(run_variants):
        print("#" * 80)
        print(f"### Run {idx+1}/{len(run_variants)}")
        print("#" * 80)
        pprint.pprint(dd)
        single_run(dd, exp_name)
    if args.archive:
        archive_experiment(exp_name)
    end_run = dt.now()
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
