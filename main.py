import argparse
import itertools
from types import SimpleNamespace
import pprint
import toml
import logging
from main_pipeline import single_run
import os

from src.data_structures.loggers import setup_run_logging

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

    args = parser.parse_args()
    return args


def generate_run_variants(base_config):
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

    run_name = setup_run_logging()

    base_config = toml.load(args.input_path)

    run_variants = generate_run_variants(base_config)

    for idx, dd in enumerate(run_variants):
        print("#" * 80)
        print(f"### Run {idx+1}/{len(run_variants)}")
        print("#" * 80)
        pprint.pprint(dd)
        ns = SimpleNamespace(**dd)
        single_run(ns, run_name)
