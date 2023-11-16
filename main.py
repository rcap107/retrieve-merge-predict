import argparse
import itertools
import logging
import os
import pprint
from datetime import datetime as dt
from types import SimpleNamespace

import toml
from sklearn.model_selection import ParameterGrid

from src.pipeline import prepare_config_dict, single_run
from src.utils.logging import (
    archive_experiment,
    get_exp_name,
    setup_run_logging,
    wrap_up_plot,
)

logger_sh = logging.getLogger("pipeline")
# console handler for info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# set formatter
ch_formatter = logging.Formatter("'%(asctime)s - %(message)s'")
ch.setFormatter(ch_formatter)
logger_sh.addHandler(ch)


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

    args = parser.parse_args()
    return args


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
    if not args.debug:
        wrap_up_plot(exp_name, base_config["run_parameters"]["task"])
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
    # finish_run()
