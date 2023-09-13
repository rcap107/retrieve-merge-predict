# %%
import argparse
import itertools
from types import SimpleNamespace
import pprint
import toml
import logging
from main_pipeline import single_run
import os
import random
import string

logger_sh = logging.getLogger("pipeline")
# console handler for info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# set formatter
ch_formatter = logging.Formatter("'%(asctime)s - %(message)s'")
ch.setFormatter(ch_formatter)
logger_sh.addHandler(ch)
# %%

if __name__ == "__main__":
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/json", exist_ok=True)

    alphabet = string.ascii_lowercase + string.digits
    run_name = "".join(random.choices(alphabet, k=8))
    os.makedirs(f"results/json/{run_name}")
    os.makedirs(f"results/logs/{run_name}")
    os.makedirs(f"results/logs/{run_name}/run_logs")
    os.makedirs(f"results/logs/{run_name}/candidate_logs")

    cfg = toml.load("config.ini")
    config_dict = cfg["DEFAULT"]
    run_sets = [k for k in cfg.keys() if k != "DEFAULT"]
    for k in run_sets:
        config_dict.update(cfg[k])
    config_dict = {k: (v if type(v) == list else [v]) for k, v in config_dict.items()}

    keys, values = zip(*config_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for dd in permutations_dicts:
        print("#" * 80)
        pprint.pprint(dd)
        ns = SimpleNamespace(**dd)
        single_run(ns, run_name)
