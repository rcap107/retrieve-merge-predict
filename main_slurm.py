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

import submitit
import toml
from joblib import Memory
from tqdm import tqdm

mem = Memory(location="__cache__", verbose=0)


# Fixing the number of polars threads for better reproducibility.
os.environ["POLARS_MAX_THREADS"] = "32"
from src.pipeline import prepare_config_dict, single_run
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
        help="Path of the experiment to recover",
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

    parser.add_argument(
        "--gpu", action="store_true", help="Whether or not to run computation on a GPU."
    )
    parser.add_argument(
        "--n-cpus",
        "-w",
        type=int,
        default=10,
        help="Number of CPUs per run of run_one.",
    )

    return parser.parse_args()


@mem.cache
def run_one(parameters, gpu=False):
    # Setup device to use GPU or not
    if gpu and torch.cuda.is_available():
        device = "cuda"
    elif gpu:
        raise RuntimeError("requested GPU run but cuda is not available.")
    else:
        device = "cpu"
    print("Using device:", device)

    # do computation and save as file.
    result = 42

    return result


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results/logs", exist_ok=True)

    start_run = dt.now()
    base_config = toml.load(args.input_path)
    run_variants = prepare_config_dict(base_config, args.debug)

    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name(debug=args.debug)

    pth_missing_runs = Path("results/logs/", exp_name, "missing_runs.pickle")

    # Submit one task per set of parameters
    executor = get_executor_marg(
        "retrieve-merge-predict",
        timeout_hour=72,
        n_cpus=args.n_cpus,
        max_parallel_tasks=10,
        gpu=args.gpu,
    )

    # Run the computation on SLURM cluster with `submitit`
    print("Submitting jobs...", end="", flush=True)
    with executor.batch():
        tasks = [
            executor.submit(single_run, (parameters, exp_name), gpu=args.gpu)
            for parameters in run_variants
        ]

    end_run = dt.now()
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
