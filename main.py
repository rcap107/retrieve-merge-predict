import argparse
import itertools
import logging
import os
import pprint
from types import SimpleNamespace

import toml

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
        "-i",
        "--input_path",
        required=True,
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


def generate_run_variants(base_config):
    # TODO: this needs updating. "DEFAULT" is a bad idea. Some arguments are not needed anymore.
    config_dict = base_config["DEFAULT"]
    run_sets = [k for k in base_config.keys() if k != "DEFAULT"]
    all_run_variants = []
    for k in run_sets:
        config_dict.update(base_config[k])
        config_dict = {
            k: (v if type(v) == list else [v]) for k, v in config_dict.items()
        }

        keys, values = zip(*config_dict.items())
        run_variants = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_run_variants += run_variants
    return all_run_variants


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/json", exist_ok=True)

    base_config = toml.load(args.input_path)
    run_variants = generate_run_variants(base_config)
    if not args.debug:
        exp_name = setup_run_logging(base_config)
    else:
        exp_name = get_exp_name()
    for idx, dd in enumerate(run_variants):
        print("#" * 80)
        print(f"### Run {idx+1}/{len(run_variants)}")
        print("#" * 80)
        pprint.pprint(dd)
        ns = SimpleNamespace(**dd)
        single_run(ns, exp_name)
    if args.archive:
        archive_experiment(exp_name)
