# %%
# %load_ext autoreload
# %autoreload 2

# %%
import json
import pickle
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import polars as pl
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.utils.logging import read_and_process, read_logs

pl.Config.set_fmt_str_lengths(150)
pl.Config.set_tbl_rows(-1)
# %%
# Defining the default configuration that will be updated by the others
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

open_data_mapping = {
    "company_employees": (
        "data/source_tables/open_data_us/company_employees-depleted_name-open_data.parquet",
        "name",
    ),
    "housing_prices": (
        "data/source_tables/open_data_us/housing_prices-depleted_County-open_data.parquet",
        "County",
    ),
    "us_elections": (
        "data/source_tables/open_data_us/us_elections-depleted_county_name-open_data.parquet",
        "county_name",
    ),
    "us_accidents_2021": (
        "data/source_tables/open_data_us/us_accidents_2021-depleted-open_data_County.parquet",
        "County",
    ),
    "us_accidents_large": (
        "data/source_tables/open_data_us/us_accidents_large-depleted-open_data_County.parquet",
        "County",
    ),
    "schools": (
        "data/source_tables/open_data_us/schools-depleted-open_data.parquet",
        "col_to_embed",
    ),
}
# %%


def configs_missing(df):
    # Configurations missing from the current file
    return df.filter(pl.col("status").is_null())


def configs_not_finished(df, group_keys):
    # Configs with fewer than 10 folds done in the current file
    return (
        df.filter(~pl.col("status").is_null())
        .group_by(group_keys)
        .agg(pl.len())
        .filter(pl.col("len") < 10)
    )


def duplicate_configs(df, group_keys):
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


def get_configs_to_review(df_config, df_results):
    group_keys = df_config.columns
    df_test = df_config.join(
        df_results.with_columns(
            base_table=pl.col("base_table").str.split("-").list.first()
        ),
        on=group_keys,
        how="left",
    )
    _cm = configs_missing(df_test)
    _cnf = configs_not_finished(df_test, group_keys)
    return (_cm, _cnf)


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

        if d["target_dl"] == "open_data_us":
            table_path, query_column = open_data_mapping[d["base_table"]]
            table_path = Path(table_path)
            up_["query_cases"]["query_column"] = query_column
        else:
            up_["query_cases"]["query_column"] = "col_to_embed"
            table_path = Path(
                "data/source_tables/yadl", f'{d["base_table"]}-yadl-depleted.parquet'
            )
        up_["query_cases"]["table_path"] = str(table_path)

        updated_configs.append(deepcopy(up_))

    print(f"Writing file config/{config_name}")
    pickle.dump(updated_configs, open(f"config/{config_name}", "wb"))


# %%
df_overall = pl.read_csv("results/master_list.csv")

# %%
cfg_path = Path("config/required_configurations/open_data_us/required_general_nn.json")

required_config = json.load(open(cfg_path, "r"))

# Given the configuration grid specified above, prepare a dataframe that contains
# all the configurations that
#  should be run
df_config = prepare_config(required_config)
group_keys = df_config.columns

_cm, _cnf = get_configs_to_review(df_config, df_overall)
configs_to_review = pl.concat([_cm.select(group_keys), _cnf.select(group_keys)])

configs_to_review
# %%
prepare_specific_configs(configs_to_review, "review-nn-open_data-general.pickle")
# %%
