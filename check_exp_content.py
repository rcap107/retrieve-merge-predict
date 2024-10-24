# %%
from pathlib import Path

import polars as pl
from sklearn.model_selection import ParameterGrid

from src.utils.logging import read_and_process, read_logs


# %%
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
    "jd_method": [
        "exact_matching",
        "minhash",
        "minhash_hybrid",
        # "starmie"
    ],
    "estimator": [
        "nojoin",
        "highest_containment",
        "full_join",
        "best_single_join",
        "stepwise_greedy_join",
    ],
    "chosen_model": [
        # "ridge",
        # "ridge_cv",
        # "catboost",
        "realmlp",
        "resnet",
        # "linear"
    ],
    "target_dl": [
        "binary_update",
        "wordnet_full",
        "wordnet_vldb_10",
        "wordnet_vldb_50",
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
    "aggregation": ["first"],
}


# %%
df_config = prepare_config(required_config)
group_keys = df_config.columns
# %%
run_ids = [
    "0667",
    "0665",
    "0638",
    "0637",
    "0636",
    "0635",
    "0671",
    # "0672",
]

base_path = "results/logs/"

dest_path = Path("results/overall")
overall_list = []

for r_path in Path(base_path).iterdir():
    r_id = str(r_path.stem).split("-")[0]
    if r_id in run_ids:
        print(r_path)
        try:
            df_raw = read_logs(exp_name=None, exp_path=r_path)
            df_raw = df_raw.fill_null(0).with_columns(
                pl.lit(0.0).alias("auc"), pl.lit(0.0).alias("f1score")
            )
            overall_list.append(df_raw)
        except pl.exceptions.SchemaError:
            print("Failed ", r_path)

df_linear = pl.read_csv("results/partial_linear.csv")
overall_list.append(df_linear)

df_overall = pl.concat(overall_list).with_columns(
    base_table=pl.col("base_table").str.split("-").list.first()
)


# %%
df_test = df_config.join(df_overall, on=group_keys, how="left")
# %%
configs_missing(df_test)
# %%
configs_not_finished(df_test)
# %%
