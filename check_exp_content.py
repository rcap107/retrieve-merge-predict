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
    pars = ParameterGrid(required_config)
    df_config = pl.from_dicts(list(pars))
    return df_config


# %%
required_config = {
    "jd_method": ["exact_matching", "minhash", "minhash_hybrid", "starmie"],
    "estimator": [
        "nojoin",
        "highest_containment",
        "full_join",
        "best_single_join",
        "stepwise_greedy_join",
    ],
    "chosen_model": [
        # "ridgecv",
        # "catboost",
        "realmlp",
        "resnet",
    ],
    "target_dl": [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        # "wordnet_vldb_50",
        # "open_data_us",
    ],
    "base_table": [
        "company_employees",
        "housing_prices",
        "us_accidents_2021",
        "us_accidents_large",
        "us_county_population",
        "us_elections",
        # "schools",
    ],
    "aggregation": [
        "first",
        # "mean",
        # "dfs"
    ],
}


# %%
df_config = prepare_config(required_config)
group_keys = df_config.columns
# %%
run_ids = [
    "0428",
    "0429",
    "0430",
    "0453",
    "0454",
    "0459",
    "0457",
    "0467",
    "0468",
    "0476",
    "0477",
    "0478",
    "0481",
    "0482",
    "0483",
    "0485",
    "0471",
    "0484",
    "0486",
    "0487",
    "0494",
    "0495",
    "0496",
    "0497",
    "0501",
    "0500",
    "0502",
    "0503",
    "0635",
    "0636",
    "0637",
    "0638",
    "0665",
    "0671",
    "0672",
    "0673",
    "0674",
    "0680",
    "0682",
    "0683",
    "0686",
]
run_ids = sorted(list(set(run_ids)))

base_path = "results/logs/"
dest_path = Path("results/overall")
overall_list = []

for r_path in tqdm(
    Path(base_path).iterdir(), total=sum(1 for _ in Path(base_path).iterdir())
):
    r_id = str(r_path.stem).split("-")[0]
    if r_id in run_ids:
        try:
            df_raw = read_logs(exp_name=None, exp_path=r_path)
            if r_id == "0673":
                df_raw = df_raw.with_columns(chosen_model=pl.lit("ridgecv"))
            overall_list.append(df_raw)
        except pl.exceptions.SchemaError:
            print("Failed ", r_path)

df_overall = pl.concat(overall_list).with_columns(
    base_table=pl.col("base_table").str.split("-").list.first()
)
df_test = df_config.join(df_overall, on=group_keys, how="left")
_cm = configs_missing(df_test)
_cnf = configs_not_finished(df_test)
configs_to_review = pl.concat([_cm.select(group_keys), _cnf.select(group_keys)])

# %%
updated_configs = []
for d in configs_to_review.to_dicts():
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

pickle.dump(updated_configs, open("config/missing_starmie_nn.pickle", "wb"))

# %%
