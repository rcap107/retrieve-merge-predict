# %%
# %load_ext autoreload
# %autoreload 2

import json

#%%
import pickle
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import polars as pl
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.utils.logging import read_and_process, read_logs

# %%
# %%
# For general use
default_config = {
    "estimators": {
        "best_single_join": {"active": True, "use_rf": False},
        "full_join": {"active": True},
        "highest_containment": {"active": True},
        "no_join": {"active": True},
        "stepwise_greedy_join": {
            "active": True,
            "budget_amount": 30,
            "budget_type": "iterations",
            "epsilon": 0.0,
            "ranking_metric": "containment",
            "use_rf": False,
        },
        "top_k_full_join": {"active": False, "top_k": 1},
    },
    "evaluation_models": {
        "catboost": {
            "iterations": 300,
            "l2_leaf_reg": 0.01,
            "od_type": "Iter",
            "od_wait": 10,
            "thread_count": 32,
        },
        "chosen_model": None,
    },
    "join_parameters": {"aggregation": "first", "join_strategy": "left"},
    "query_cases": {
        "data_lake": None,
        "join_discovery_method": None,
        "query_column": None,
        "table_path": None,
        "top_k": 30,
    },
    "run_parameters": {
        "debug": False,
        "n_splits": 10,
        "split_kind": "group_shuffle",
        "task": "regression",
        "test_size": 0.2,
    },
}


def configs_missing(df):
    # Configurations missing from the current file
    return df.filter(pl.col("status").is_null())


def configs_not_finished(df):
    # Configs with fewer than 10 folds done in the current file
    return (
        df.filter(~pl.col("status").is_null())
        .group_by(group_keys)
        .agg(pl.len())
        .filter(pl.col("len") < 10)
    )


def duplicate_configs(df):
    # Configs with more than 10 folds done in the current file (duplicates?)
    return (
        df.filter(~pl.col("status").is_null())
        .group_by(group_keys)
        .agg(pl.len())
        .filter(pl.col("len") > 10)
    )


def prepare_config(config_dict):
    pars = ParameterGrid(config_dict)
    df_config = pl.from_dicts(list(pars))
    return df_config


# %%
cfg_path = Path("config/required_configurations/yadl/required_general.json")

required_config = json.load(open(cfg_path, "r"))

# Given the configuration grid specified above, prepare a dataframe that contains
# all the configurations that should be run
df_config = prepare_config(required_config)
group_keys = df_config.columns
#%%
df_overall = pl.read_csv("results/master_list.csv")

df_test = df_config.join(df_overall, on=group_keys, how="left")
_cm = configs_missing(df_test)
_cnf = configs_not_finished(df_test)
configs_to_review = pl.concat([_cm.select(group_keys), _cnf.select(group_keys)])

# %%
def prepare_specific_configs(
    cfg_to_review,
    config_name,
):
    updated_configs = []
    for d in cfg_to_review.to_dicts():
        up_ = dict(default_config)
        up_["evaluation_models"]["chosen_model"] = d["chosen_model"]
        up_["query_cases"]["data_lake"] = d["target_dl"]
        up_["query_cases"]["join_discovery_method"] = d["jd_method"]
        up_["query_cases"]["query_column"] = "col_to_embed"

        table_path = Path(
            "data/source_tables/yadl", f'{d["base_table"]}-yadl-depleted.parquet'
        )
        up_["query_cases"]["table_path"] = table_path

        updated_configs.append(deepcopy(up_))

    pickle.dump(updated_configs, open(f"config/{config_mame}", "wb"))
