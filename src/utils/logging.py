import json
import os
import random
import string
import tarfile
from pathlib import Path

import polars as pl

import src.utils.plotting as plotting

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


def wrap_up_plot(exp_name, task="regression", variable_of_interest=None):
    """Prepare and save the plots relevant to the task under consideration.
    If the task is `regression`, plot `r2score`, if the task is `classification`,
    plot `f1score`.

    Args:
        exp_name (str): Name of the current experiment.
        task (str, optional): Task under consideration, either `regression` or
        `classification`. Defaults to "regression".
    """
    df_raw = read_logs(exp_name=exp_name)

    if task == "regression":
        current_score = "r2score"
    else:
        current_score = "f1score"

    path_target_run = Path("results/logs/", exp_name)

    if variable_of_interest is not None:
        for gname, group in df_raw.group_by(variable_of_interest):
            for case in [current_score, "time_run"]:
                path_plot = Path(path_target_run, "plots", f"{gname}_{case}.png")
                ax = plotting.base_barplot(group.to_pandas(), result_variable=case)
                ax.savefig(path_plot)

            path_plot = Path(
                path_target_run, "plots", f"{gname}_scatter_time_{current_score}.png"
            )
            ax = plotting.base_relplot(group.to_pandas(), y_variable=current_score)
            ax.savefig(path_plot)

    else:
        for case in [current_score, "time_run"]:
            path_plot = Path(path_target_run, "plots", f"overall_{case}.png")
            ax = plotting.base_barplot(df_raw.to_pandas(), result_variable=case)
            ax.savefig(path_plot)

        path_plot = Path(path_target_run, "plots", f"scatter_time_{current_score}.png")
        ax = plotting.base_relplot(df_raw.to_pandas(), y_variable=current_score)
        ax.savefig(path_plot)


def prepare_data_for_plotting(df: pl.DataFrame) -> pl.DataFrame:
    max_diff = df.select(pl.col("difference").abs().max()).item()
    df = df.with_columns((pl.col("difference") / max_diff).alias("scaled_diff"))
    return df


def read_and_process(df_results):
    df_ = df_results.select(
        pl.col(
            [
                "scenario_id",
                "fold_id",
                "target_dl",
                "jd_method",
                "base_table",
                "estimator",
                "chosen_model",
                "aggregation",
                "r2score",
                "time_fit",
                "time_predict",
                "time_run",
            ]
        )
    ).with_columns(
        (
            pl.col("base_table").str.split("-").list.first() + "-" + pl.col("target_dl")
        ).alias("case")
    )
    # df_ = df_.group_by(
    #     [_ for _ in GROUPING_KEYS if _ != "fold_id"]
    # ).map_groups(lambda x: x.with_row_count("fold_id"))

    joined = df_.join(
        df_.filter(pl.col("estimator") == "nojoin"),
        on=GROUPING_KEYS,
        how="left",
    ).with_columns((pl.col("r2score") - pl.col("r2score_right")).alias("difference"))

    projection = [
        "fold_id",
        "target_dl",
        "jd_method",
        "base_table",
        "case",
        "estimator",
        "chosen_model",
        "aggregation",
        "r2score",
        "time_fit",
        "time_predict",
        "time_run",
        "difference",
    ]
    joined = joined.select(projection)

    results_full = joined.filter(~pl.col("base_table").str.contains("depleted"))
    results_depleted = joined.filter(pl.col("base_table").str.contains("depleted"))

    results_full = prepare_data_for_plotting(results_full)
    results_depleted = prepare_data_for_plotting(results_depleted)

    return results_full, results_depleted
