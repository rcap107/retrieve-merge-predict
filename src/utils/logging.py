import json
import os
import random
import string
import tarfile
from pathlib import Path

import polars as pl

RUN_ID_PATH = Path("results/run_id")
SCENARIO_ID_PATH = Path("results/scenario_id")

GROUPING_KEYS = [
    "jd_method",
    "estimator",
    "chosen_model",
    "target_dl",
    "base_table",
    "aggregation",
    "fold_id",
]

HEADER_RUN_LOGFILE = [
    "scenario_id",
    "status",
    "target_dl",
    "jd_method",
    "base_table",
    "query_column",
    "estimator",
    "aggregation",
    "chosen_model",
    "fold_id",
    "time_fit",
    "time_predict",
    "time_run",
    "time_prepare",
    "time_model_train",
    "time_join_train",
    "time_model_predict",
    "time_join_predict",
    "peak_fit",
    "peak_predict",
    "peak_test",
    "r2score",
    "rmse",
    "f1score",
    "auc",
    "n_cols",
    "budget_type",
    "budget_amount",
    "epsilon",
]


def get_exp_name(debug=False):
    alphabet = string.ascii_lowercase + string.digits
    random_slug = "".join(random.choices(alphabet, k=8))
    scenario_id = read_and_update_scenario_id(debug=debug)

    exp_name = f"{scenario_id:04d}-{random_slug}"
    return exp_name


def read_logs(exp_name=None, exp_path=None):
    if exp_name is not None:
        path_target_run = Path("results/logs/", exp_name)
    else:
        path_target_run = Path(exp_path)
    path_agg_logs = Path(path_target_run, "run_logs")

    logs = []
    for f in path_agg_logs.glob("*.log"):
        logs.append(pl.read_csv(f))
    df_agg = pl.concat(logs)

    return df_agg


def setup_run_logging(setup_config=None):
    exp_name = get_exp_name()
    os.makedirs(f"results/logs/{exp_name}")
    os.makedirs(f"results/logs/{exp_name}/json")
    os.makedirs(f"results/logs/{exp_name}/json/failed")
    os.makedirs(f"results/logs/{exp_name}/plots")
    os.makedirs(f"results/logs/{exp_name}/run_logs")
    os.makedirs(f"results/logs/{exp_name}/raw_logs")

    if setup_config is not None:
        with open(f"results/logs/{exp_name}/{exp_name}.cfg", "w") as fp:
            json.dump(setup_config, fp, indent=2)

    return exp_name


def read_and_update_scenario_id(exp_name=None, debug=False):
    if debug:
        return 0
    if exp_name is None:
        scenario_id_path = SCENARIO_ID_PATH
    else:
        scenario_id_path = Path(f"results/logs/{exp_name}/scenario_id")

    if scenario_id_path.exists():
        with open(scenario_id_path, "r") as fp:
            last_scenario_id = fp.read().strip()
            if len(last_scenario_id) != 0:
                try:
                    scenario_id = int(last_scenario_id) + 1
                except ValueError:
                    raise ValueError(
                        f"Scenario ID {last_scenario_id} is not a positive integer. "
                    )
                if scenario_id < 0:
                    raise ValueError(
                        f"Scenario ID {scenario_id} is not a positive integer. "
                    )
            else:
                scenario_id = 0
        with open(scenario_id_path, "w") as fp:
            fp.write(f"{scenario_id:04d}")
    else:
        scenario_id = 0
        with open(scenario_id_path, "w") as fp:
            fp.write(f"{scenario_id:04d}")
    return scenario_id


def read_scenario_id():
    scenario_id_path = SCENARIO_ID_PATH
    if scenario_id_path.exists():
        with open(scenario_id_path, "r") as fp:
            last_scenario_id = fp.read().strip()
            if len(last_scenario_id) != 0:
                try:
                    scenario_id = int(last_scenario_id) + 1
                except ValueError:
                    raise ValueError(
                        f"Scenario ID {last_scenario_id} is not a positive integer. "
                    )
                if scenario_id < 0:
                    raise ValueError(
                        f"Scenario ID {scenario_id} is not a positive integer. "
                    )
                return last_scenario_id
            else:
                return 0
    else:
        raise IOError(f"SCENARIO_ID_PATH {scenario_id_path} not found.")


def archive_experiment(exp_name):
    archive_path = Path("results/archives")
    results_path = Path(f"results/logs/{exp_name}")
    os.makedirs(archive_path, exist_ok=True)
    archive_name = exp_name + ".tar"

    with tarfile.open(Path(archive_path, archive_name), mode="x") as tar:
        tar.add(results_path, arcname=exp_name)


def prepare_data_for_plotting(df: pl.DataFrame) -> pl.DataFrame:
    max_diff = df.select(pl.col("difference").abs().max()).item()
    df = df.with_columns((pl.col("difference") / max_diff).alias("scaled_diff"))
    return df


def read_and_process(df_results):
    """This function processes all runs in a fixed way to have consistent results.

    Args:
        df_results (pl.DataFrame): Dataframe that contains the results.

    Returns:
        results: Prepared results
    """
    keep_cases = [
        "us_accidents_2021-yadl-depleted",
        "housing_prices-yadl-depleted",
        "company_employees-yadl-depleted",
        "us_accidents_large-yadl-depleted",
        "us_county_population-yadl-depleted",
        "us_elections-depleted_county_name-open_data",
        "company_employees-depleted_name-open_data",
        "schools-depleted-open_data",
        "housing_prices-depleted_County-open_data",
        "us_elections-yadl-depleted",
        "schools-depleted-open_data",
        # "movies_large-yadl-depleted",
        # "movies_large-depleted-open_data",
        "us_accidents_2021-depleted-open_data_County",
        "us_accidents_large-depleted-open_data_County",
    ]
    df_results = df_results.filter(pl.col("base_table").is_in(keep_cases))

    df_ = df_results.select(
        pl.col(
            [
                "scenario_id",
                "fold_id",
                "target_dl",
                "jd_method",
                "base_table",
                "query_column",
                "estimator",
                "chosen_model",
                "aggregation",
                "r2score",
                "auc",
                "time_fit",
                "time_predict",
                "time_run",
                "peak_fit",
                "peak_predict",
            ]
        )
    ).with_columns(
        case=(
            pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
        ),
        y=pl.when(pl.col("auc") > 0).then(pl.col("auc")).otherwise(pl.col("r2score")),
    )

    joined = df_.join(
        df_.filter(pl.col("estimator") == "nojoin"),
        on=[_ for _ in GROUPING_KEYS if _ != "estimator"],
        how="left",
    ).with_columns((pl.col("y") - pl.col("y_right")).alias("difference"))

    projection = [
        "fold_id",
        "target_dl",
        "jd_method",
        "base_table",
        "query_column",
        "case",
        "estimator",
        "chosen_model",
        "aggregation",
        "y",
        "time_fit",
        "time_predict",
        "time_run",
        "peak_fit",
        "peak_predict",
        "difference",
    ]
    joined = joined.select(projection)

    _results = joined.filter(pl.col("base_table").str.contains("depleted"))
    _results = prepare_data_for_plotting(_results)

    return _results
