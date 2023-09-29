import json
import os
import random
import string
import tarfile
from pathlib import Path

import polars as pl

RUN_ID_PATH = Path("results/run_id")
SCENARIO_ID_PATH = Path("results/scenario_id")

HEADER_LOGFILE = [
    "scenario_id",
    "run_id",
    "status",
    "target_dl",
    "git_hash",
    "index_name",
    "base_table",
    "candidate_table",
    "iterations",
    "join_strategy",
    "aggregation",
    "fold_id",
    "time_train",
    "time_eval",
    "time_join",
    "time_eval_join",
    "best_candidate_hash",
    "n_cols",
    "r2score",
    "avg_r2",
    "std_r2",
    "tree_count",
    "best_iteration",
]


def get_exp_name(debug=False):
    alphabet = string.ascii_lowercase + string.digits
    random_slug = "".join(random.choices(alphabet, k=8))
    scenario_id = read_and_update_scenario_id(debug=debug)

    exp_name = f"{scenario_id:04d}-{random_slug}"
    return exp_name


def read_log_dir(exp_name):
    pass


def read_log_tar(exp_name):
    pass


def read_logs(exp_name):
    path_target_run = Path("results/logs/", exp_name)
    path_raw_logs = Path(path_target_run, "raw_logs")
    path_agg_logs = Path(path_target_run, "run_logs")

    logs = []
    for f in path_raw_logs.glob("*.log"):
        logs.append(pl.read_csv(f))
    df_raw = pl.concat(logs)

    logs = []
    for f in path_agg_logs.glob("*.log"):
        logs.append(pl.read_csv(f))
    df_agg = pl.concat(logs)

    return df_raw, df_agg


def setup_run_logging(setup_config=None):
    exp_name = get_exp_name()
    os.makedirs(f"results/logs/{exp_name}")
    os.makedirs(f"results/logs/{exp_name}/json")
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
