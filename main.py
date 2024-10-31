"""
Main entrypoint for the benchmarking code.
"""

import argparse
import json
import os
import pickle
import pprint
from collections import deque
from datetime import datetime as dt
from pathlib import Path

import toml
from tqdm import tqdm

# Fixing the number of polars threads for better reproducibility.
os.environ["POLARS_MAX_THREADS"] = "32"
from src.pipeline import prepare_config_dict, prepare_specific_configs, single_run
from src.utils.logging import archive_experiment, get_exp_name, setup_run_logging


def parse_args():
    """Parse arguments on the command line.

    Returns:
        argparse.Namespace: Arguments parsed on the command line.
    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--input_path",
        action="store",
        default=None,
        help="Path of the config file to be used.",
        type=argparse.FileType("r"),
    )

    group.add_argument(
        "--recovery_path",
        action="store",
        default=None,
        help="Path of the experiment to recover.",
        type=Path,
    )

    group.add_argument(
        "--selected_config",
        action="store",
        default=None,
        help="Path to the pickle of specific configurations to run.",
        type=Path,
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

    if args.recovery_path is not None:
        # If args.recovery_path is provided, the script will look for the
        # missing_runs.pickle file in the given path and
        # try to reboot a run from there.
        if args.recovery_path.exists():
            pth = args.recovery_path
            missing_runs_path = Path(pth, "missing_runs.pickle")
            missing_runs_config = Path(pth, pth.stem + ".cfg")
            with open(missing_runs_config, "r") as fp:
                base_config = json.load(fp)
            run_variants = pickle.load(open(missing_runs_path, "rb"))
        else:
            raise IOError(f"File {args.recovery_path} not found.")
    elif args.selected_config is not None:
        # Using a specific set of configurations
        if args.selected_config.exists():
            run_variants = prepare_specific_configs(args.selected_config)
            base_config = run_variants
        else:
            raise IOError(f"File {args.selected_config} not found.")
    else:
        # No recovery, simply read a toml file from the given input path.
        base_config = toml.load(args.input_path)
        run_variants = prepare_config_dict(base_config, args.debug)

    # Using a queue for better convenience when preparing missing runs.
    run_queue = deque(run_variants)

    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name(debug=args.debug)

    pth_missing_runs = Path("results/logs/", exp_name, "missing_runs.pickle")

    # Overall progress bar for all the variants in this batch of experiments.
    progress_bar = tqdm(total=len(run_variants), position=0, desc="Overall progress: ")
    while len(run_queue) > 0:
        this_config = run_queue.pop()
        # For each variant, overwrite the missing_runs.pickle file with the current missing runs
        if not args.debug:
            with open(pth_missing_runs, "wb") as fp:
                pickle.dump(list(run_queue), fp)
        pprint_config = pprint.pformat(this_config)
        tqdm.write(pprint_config)
        single_run(this_config, exp_name)
        progress_bar.update(1)
    progress_bar.close()

    if args.archive:
        archive_experiment(exp_name)
    end_run = dt.now()
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
