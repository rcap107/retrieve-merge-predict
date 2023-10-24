import argparse
import itertools
import logging
import os
import pprint
from datetime import datetime as dt
from types import SimpleNamespace

import toml
from sklearn.model_selection import ParameterGrid

from main_pipeline import single_run
from src.utils.logging import archive_experiment, get_exp_name, setup_run_logging

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


def generate_run_variants(base_config, debug=False):
    base_config["input_data"]["debug"] = debug
    sections = list(base_config.keys())
    config_dict = {}
    for k in base_config:
        config_dict.update(base_config[k])
    config_dict = {k: (v if type(v) == list else [v]) for k, v in config_dict.items()}
    grid = list(ParameterGrid(config_dict))

    return grid


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results/logs", exist_ok=True)

    start_run = dt.now()
    base_config = toml.load(args.input_path)
    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name(debug=args.debug)
    run_variants = generate_run_variants(base_config, debug=args.debug)
    for idx, dd in enumerate(run_variants):
        print("#" * 80)
        print(f"### Run {idx+1}/{len(run_variants)}")
        print("#" * 80)
        pprint.pprint(dd)
        ns = SimpleNamespace(**dd)
        single_run(ns, exp_name)
    if args.archive:
        archive_experiment(exp_name)
    end_run = dt.now()
    run_duration = end_run - start_run
    print(f"Run duration: {run_duration.total_seconds():.2f} seconds")
    # finish_run()
