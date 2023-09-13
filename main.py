# %%
import argparse
from main_pipeline import single_run
from types import SimpleNamespace
import itertools
import toml

# %%
cfg = toml.load("config.ini")
config_dict = cfg["DEFAULT"]
run_sets = [k for k in cfg.keys() if k != "DEFAULT"]
for k in run_sets:
    config_dict.update(cfg[k])
# %%
config_dict = {k: (v if type(v) == list else [v]) for k, v in config_dict.items()}

keys, values = zip(*config_dict.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
# %%
for dd in permutations_dicts:
    ns = SimpleNamespace(**dd)

# %%
single_run(ns)
# %%
