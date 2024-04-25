import os

os.environ["POLARS_MAX_THREADS"] = "32"

import argparse
import json
import pickle
import pprint
from collections import deque
from datetime import datetime as dt
from pathlib import Path

import toml
from tqdm import tqdm

from src.pipeline import prepare_config_dict, single_run
from src.utils.logging import archive_experiment, get_exp_name, setup_run_logging


def parse_args():
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results/logs", exist_ok=True)

    start_run = dt.now()

    if args.recovery_path is not None:
        if args.recovery_path.exists():
            pth = args.recovery_path
            missing_runs_path = Path(pth, "missing_runs.pickle")
            missing_runs_config = Path(pth, pth.stem + ".cfg")
            with open(missing_runs_config, "r") as fp:
                base_config = json.load(fp)
            run_variants = pickle.load(open(missing_runs_path, "rb"))
        else:
            raise IOError(f"File {args.recovery_path} not found.")
    else:
        base_config = toml.load(args.input_path)
        run_variants = prepare_config_dict(base_config, args.debug)

    run_queue = deque(run_variants)

    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name(debug=args.debug)

    pth_missing_runs = Path("results/logs/", exp_name, "missing_runs.pickle")

    progress_bar = tqdm(total=len(run_variants), position=2)
    while len(run_queue) > 0:
        this_config = run_queue.pop()
        with open(pth_missing_runs, "wb") as fp:
            pickle.dump(list(run_queue), fp)
        pprint.pprint(this_config)
        single_run(this_config, exp_name)
        progress_bar.update(1)
    progress_bar.close()

    if args.archive:
        archive_experiment(exp_name)
    end_run = dt.now()
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
