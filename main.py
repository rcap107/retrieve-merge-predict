# %%
import argparse
from main_pipeline import single_run
from types import SimpleNamespace

# %%
d = {
    "yadl_version": [""],
    "source_table_path": [""],
    "query_column": [""],
    "sample_size": [""],
    "selected_indices": ["minhash"],
    "sampling_seed": [42],
    "query_result_path": [""],
    "iterations": [1000],
    "join_strategy": ["left"],
    "aggregation": ["first"],
    "n_splits": [5],
    "top_k": [50],
    "dry_run": [False],
    "feature_selection": [False],
    "model_selection": [False],
    "n_jobs": [1],
    "cuda": [False],
}

# %%
d["model_selection"] = [True, False]
d["aggregation"] = ["first", "mean", "dfs"]
# %%
import itertools

keys, values = zip(*d.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

# %%
for dd in permutations_dicts:
    ns = SimpleNamespace(**dd)

# %%
import toml

cfg = toml.load("config.ini")
defaults = cfg["DEFAULT"]
defaults.update(cfg["wordnet"])
# %%
defaults = {k: (v if type(v) == list else [v]) for k, v in defaults.items()}
import itertools

keys, values = zip(*defaults.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
# %%
