import copy
import csv
import datetime as dt
import json
import logging
from pathlib import Path
from time import process_time

import polars as pl
from sklearn.metrics import f1_score, r2_score, roc_auc_score, root_mean_squared_error
from tqdm import tqdm

import src.utils.logging as log
from src.utils.logging import HEADER_RUN_LOGFILE

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="a",
    filename="data/logging.txt",
    level=logging.DEBUG,
)


class ScenarioLogger:
    def __init__(
        self,
        base_table_name,
        git_hash,
        run_config,
        exp_name=None,
        debug=False,
    ) -> None:
        self.timestamps = {
            "start_process": dt.datetime.now(),
            "end_process": 0,
            "start_load_index": 0,
            "end_load_index": 0,
            "start_querying": 0,
            "end_querying": 0,
            "start_evaluation": 0,
            "end_evaluation": 0,
        }
        self.exp_name = exp_name
        self.scenario_id = log.read_and_update_scenario_id(exp_name, debug=debug)
        self.prepare_logger(exp_name)

        self.estim_parameters = run_config["estimators"]
        self.model_parameters = run_config["evaluation_models"]
        self.join_parameters = run_config["join_parameters"]
        self.run_parameters = run_config["run_parameters"]
        self.query_info = run_config["query_cases"]

        self.task = self.run_parameters["task"]
        self.run_id = 0
        self.start_timestamp = None
        self.end_timestamp = None
        self.chosen_model = self.model_parameters["chosen_model"]
        self.jd_method = self.query_info["join_discovery_method"]
        self.base_table_name = base_table_name
        self.git_hash = git_hash
        if self.chosen_model == "catboost":
            self.iterations = self.model_parameters["catboost"]["iterations"]
        else:
            self.iterations = 0
        self.aggregation = self.join_parameters["aggregation"]
        self.target_dl = self.query_info["data_lake"]
        self.n_splits = self.run_parameters["n_splits"]
        self.top_k = self.query_info["top_k"]
        self.results = None
        self.process_time = 0
        self.status = None
        self.exception_name = None
        self.additional_info = None
        self.debug = debug

    def prepare_logger(self, run_name=None):
        self.path_run_logs = f"results/logs/{run_name}/run_logs/{self.scenario_id}.log"
        self.path_raw_logs = f"results/logs/{run_name}/raw_logs/{self.scenario_id}.log"

    def add_timestamp(self, which_ts):
        self.timestamps[which_ts] = dt.datetime.now()

    def add_process_time(self):
        self.process_time = process_time()

    def get_parameters(self):
        return {
            "base_table": self.base_table_name,
            "jd_method": self.jd_method,
            "iterations": self.iterations,
            "aggregation": self.aggregation,
            "target_dl": self.target_dl,
            "n_splits": self.n_splits,
            "top_k": self.top_k,
        }

    def set_results(self, results: pl.DataFrame):
        self.results = results

    def get_next_run_id(self):
        self.run_id += 1
        return self.run_id

    def to_string(self):
        str_res = ",".join(
            map(
                str,
                [
                    self.scenario_id,
                    self.git_hash,
                    self.base_table_name,
                    self.jd_method,
                    self.iterations,
                    self.aggregation,
                    self.target_dl,
                    self.n_splits,
                    self.top_k,
                ],
            )
        )
        str_res += ","
        for ts in self.timestamps.values():
            str_res += str(ts) + ","

        return str_res.rstrip(",")

    def print_results(self):
        self.tqdm_print()
        if self.task == "regression":
            summary = self.results.group_by(["estimator"]).agg(
                pl.mean("r2"), pl.mean("rmse")
            )
        else:
            summary = self.results.group_by(["estimator"]).agg(
                pl.mean("f1"), pl.mean("auc")
            )
        print(summary)

    def pretty_print(self):
        print(f"Scenario ID: {self.scenario_id}")
        print(f"Run name: {self.exp_name}")
        print(f"Base table: {self.base_table_name}")
        print(f"DL Variant: {self.target_dl}")
        print(f"Retrieval method: {self.jd_method}")
        print(f"Aggregation: {self.aggregation}")
        print(f"ML model: {self.chosen_model}")

    def tqdm_print(self):
        tqdm.write(f"Scenario ID: {self.scenario_id}")
        tqdm.write(f"Run name: {self.exp_name}")
        tqdm.write(f"Base table: {self.base_table_name}")
        tqdm.write(f"DL Variant: {self.target_dl}")
        tqdm.write(f"Retrieval method: {self.jd_method}")
        tqdm.write(f"Aggregation: {self.aggregation}")
        tqdm.write(f"ML model: {self.chosen_model}")

    def write_to_log(self, out_path):
        if Path(out_path).parent.exists():
            with open(out_path, "a") as fp:
                fp.write(self.to_string() + "\n")

    def set_status(self, status, exception_name=None):
        """Set run status for logging.

        Args:
            status (str): Status to use.
        """
        self.status = status
        if exception_name is not None:
            self.exception_name = exception_name
        else:
            self.exception_name = ""

    def write_to_json(self, root_path="results/logs/"):
        if self.debug:
            print("ScenarioLogger is in debug mode, no logs will be written.")
            return None
        res_dict = copy.deepcopy(vars(self))
        if self.results is not None:
            results = self.results.clone()
            res_dict["results"] = results.to_dicts()
        else:
            res_dict["results"] = None
        res_dict["timestamps"] = {
            k: v.isoformat()
            for k, v in res_dict["timestamps"].items()
            if isinstance(v, dt.datetime)
        }
        if Path(root_path).exists():
            dest_path = Path(root_path, self.exp_name, "json")
            if self.status == "SUCCESS":
                dest_path = Path(dest_path, f"{self.scenario_id}.json")
            elif self.status == "FAILURE":
                dest_path = Path(dest_path, "failed", f"{self.scenario_id}.json")

            with open(dest_path, "w") as fp:
                json.dump(res_dict, fp, indent=2)
        else:
            raise IOError(f"Invalid path {root_path}")

    def write_summary_plot(self, root_path):
        import matplotlib.pyplot as plt
        import seaborn as sns

        keys = [
            "exp_name",
            "scenario_id",
            "chosen_model",
            "base_table_name",
            "git_hash",
            "iterations",
            "target_dl",
            "aggregation",
        ]
        dict_print = {k: v for k, v in vars(self).items() if k in keys}
        annotation = json.dumps(dict_print, indent=2)
        fig = plt.figure()
        axs = fig.subplots(nrows=2)
        if self.task == "regression":
            x_value = "r2"
        else:
            x_value = "f1"
        sns.boxplot(
            data=self.results.to_pandas(),
            y="estimator",
            x=x_value,
            ax=axs[0],
            orient="h",
        )
        axs[1].text(0, 0, annotation)
        axs[1].axis("off")
        plt.tight_layout()
        fig_path = Path(root_path, self.exp_name, "plots", f"{self.scenario_id}.png")
        fig.savefig(fig_path)

    def finish_run(self, root_path="results/logs/"):
        if not self.debug and self.status == "SUCCESS":
            # self.write_summary_plot(root_path)
            self.write_to_json(root_path)
        elif self.status == "FAILURE":
            print("Run failed.")
            self.write_to_json(root_path)
        else:
            print("ScenarioLogger is in debugging mode. No files will be created.")


class RunLogger:
    def __init__(
        self,
        scenario_logger: ScenarioLogger,
        additional_parameters: dict,
        fold_id: int = None,
    ):
        # TODO: rewrite with __getitem__ instead
        self.scenario_id = scenario_logger.scenario_id
        self.path_run_logs = scenario_logger.path_run_logs
        self.path_raw_logs = scenario_logger.path_raw_logs
        self.run_id = scenario_logger.get_next_run_id()
        self.task = scenario_logger.task
        self.debug = scenario_logger.debug
        self.status = None
        self.timestamps = {}
        self.durations = {
            "time_run": "",
            "time_fit": "",
            "time_predict": "",
        }

        self.parameters = self.get_parameters(scenario_logger, additional_parameters)
        self.parameters["fold_id"] = fold_id
        self.results = {}
        self.additional_info = {
            "best_candidate_hash": None,
            "left_on": None,
            "right_on": None,
            "joined_columns": None,
        }

        self.memory_usage = {
            "fit": None,
            "predict": None,
            "test": None,
            "peak_fit": None,
            "peak_predict": None,
            "peak_test": None,
        }

        self.mark_time("run")

    def get_parameters(self, scenario_logger: ScenarioLogger, additional_parameters):
        parameters = {
            "base_table": scenario_logger.base_table_name,
            "candidate_table": "",
            "left_on": "",
            "right_on": "",
            "git_hash": scenario_logger.git_hash,
            "iterations": scenario_logger.iterations,
            "aggregation": scenario_logger.aggregation,
            "target_dl": scenario_logger.target_dl,
            "jd_method": scenario_logger.jd_method,
            "fold_id": "",
            "query_column": scenario_logger.query_info["query_column"],
        }
        if additional_parameters is not None:
            parameters.update(additional_parameters)

        return parameters

    def update_parameters(self, additional_parameters):
        self.parameters.update(additional_parameters)

    def update_timestamps(self, additional_timestamps=None):
        if additional_timestamps is not None:
            self.timestamps.update(additional_timestamps)

    def update_durations(self, additional_durations=None):
        if additional_durations is not None:
            self.durations.update(additional_durations)

    def set_run_status(self, status: str):
        """Set run status for logging.

        Args:
            status (str): Status to use.
        """
        self.status = status

    def start_time(self, label: str, cumulative: bool = False):
        """Wrapper around the `mark_time` function for better clarity.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.
        """
        return self.mark_time(label, cumulative)

    def end_time(self, label: str, cumulative: bool = False):
        if label not in self.timestamps:
            raise KeyError(f"Label {label} was not found.")
        return self.mark_time(label, cumulative)

    def mark_time(self, label: str, cumulative: bool = False):
        """Given a `label`, add a new timestamp if `label` isn't found, otherwise
        mark the end of the timestamp and add a new duration.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.

        """
        if label not in self.timestamps:
            self.timestamps[label] = [dt.datetime.now(), None]
            self.durations["time_" + label] = 0
        else:
            self.timestamps[label][1] = dt.datetime.now()
            this_segment = self.timestamps[label]
            if cumulative:
                self.durations["time_" + label] += (
                    this_segment[1] - this_segment[0]
                ).total_seconds()
            else:
                self.durations["time_" + label] = (
                    this_segment[1] - this_segment[0]
                ).total_seconds()

    def get_time(self, label: str):
        """Retrieve a time according to the given label.

        Args:
            label (str): Label of the timestamp to be retrieved.
        Returns:
            _type_: Retrieved timestamp.
        """
        if label in self.timestamps:
            return self.timestamps[label]
        else:
            raise KeyError(f"Label {label} not found in timestamps.")

    def get_duration(self, label: str):
        """Retrieve a duration according to the given label.

        Args:
            label (str): Label of the duration to be retrieved.
        Returns:
            _type_: Retrieved duration.

        Raises:
            KeyError if the provided label is not found.
        """
        if label in self.durations:
            return self.durations[label]
        else:
            raise KeyError(f"Label {label} not found in durations.")

    def mark_memory(self, mem_usage: list, label: str):
        """Record the memory usage for a given section of the code.

        Args:
            mem_usage (list): List containing the memory usage and timestamps.
            label (str): One of "fit", "predict", "test".

        Raises:
            KeyError: Raise KeyError if the label is not correct.
        """
        if label in self.memory_usage:
            self.memory_usage[label] = mem_usage
            self.memory_usage[f"peak_{label}"] = max(
                _[0] for _ in self.memory_usage[label]
            )
        else:
            raise KeyError(f"Label {label} not found in mem_usage.")

    def measure_results(self, y_true, y_pred):
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_pandas()
        if isinstance(y_pred, pl.Series):
            y_pred = y_pred.to_pandas()

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("The provided vectors have inconsistent shapes.")

        if self.task == "regression":
            self.results["r2"] = r2_score(y_true, y_pred)
            self.results["rmse"] = root_mean_squared_error(y_true, y_pred)
        elif self.task == "classification":
            self.results["f1"] = f1_score(y_true, y_pred)
            self.results["auc"] = roc_auc_score(y_true, y_pred)
        else:
            raise ValueError()
        return self.results

    def set_additional_info(self, info_dict: dict):
        self.additional_info.update(info_dict)

    def to_dict(self):
        values = [
            self.scenario_id,
            self.status,
            self.parameters["target_dl"],
            self.parameters["jd_method"],
            self.parameters["base_table"],
            self.parameters["query_column"],
            self.parameters["estimator"],
            self.parameters["aggregation"],
            self.parameters["chosen_model"],
            self.parameters.get("fold_id", ""),
            self.durations.get("time_fit", ""),
            self.durations.get("time_predict", ""),
            self.durations.get("time_run", ""),
            self.durations.get("time_prepare", ""),
            self.durations.get("time_model_train", ""),
            self.durations.get("time_join_train", ""),
            self.durations.get("time_model_predict", ""),
            self.durations.get("time_join_predict", ""),
            self.memory_usage.get("peak_fit", ""),
            self.memory_usage.get("peak_predict", ""),
            self.memory_usage.get("peak_test", ""),
            self.results.get("r2", ""),
            self.results.get("rmse", ""),
            self.results.get("f1", ""),
            self.results.get("auc", ""),
            self.additional_info.get("joined_columns", ""),
            self.additional_info.get("budget_type", ""),
            self.additional_info.get("budget_amount", ""),
            self.additional_info.get("epsilon", ""),
        ]

        return dict(zip(HEADER_RUN_LOGFILE, values))

    def to_str(self):
        res_str = ",".join(
            map(
                str,
                self.to_dict().values(),
            )
        )
        return res_str

    def to_raw_log_file(self):
        self.to_logfile(self.path_raw_logs)

    def to_run_log_file(self):
        self.to_logfile(self.path_run_logs)

    def to_logfile(self, path_logfile):
        if not self.debug:
            if Path(path_logfile).exists():
                with open(path_logfile, "a") as fp:
                    writer = csv.DictWriter(fp, fieldnames=HEADER_RUN_LOGFILE)
                    writer.writerow(self.to_dict())
            else:
                with open(path_logfile, "w") as fp:
                    writer = csv.DictWriter(fp, fieldnames=HEADER_RUN_LOGFILE)
                    writer.writeheader()
                    writer.writerow(self.to_dict())
        else:
            print(json.dumps(self.to_dict(), indent=2))


class SimpleIndexLogger:
    def __init__(
        self,
        index_name: str,
        step: str,
        data_lake_version: str,
        index_parameters: dict = None,
        query_parameters: dict = None,
        log_path: str | Path = "results/index_logging.txt",
    ) -> None:
        self.exp_name = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        self.log_path = log_path
        self.index_name = index_name
        self.data_lake_version = data_lake_version
        self.step = step
        self.index_parameters = index_parameters if index_parameters is not None else {}
        self.query_parameters = query_parameters if query_parameters is not None else {}

        self.timestamps = {}
        self.durations = {}

        self.memory_usage = {
            "create": None,
            "query": None,
            "peak_create": None,
            "peak_query": None,
        }

        self.query_results = {"n_candidates": 0}

        self.header = [
            "data_lake_version",
            "index_name",
            "n_jobs",
            "base_table",
            "query_column",
            "step",
            "time_create",
            "time_save",
            "time_load",
            "time_query",
            "peak_create",
            "peak_query",
            "n_candidates",
        ]

    def update_query_parameters(self, query_table, query_column):
        self.query_parameters["base_table"] = query_table
        self.query_parameters["query_column"] = query_column

    def start_time(self, label: str, cumulative: bool = False):
        """Wrapper around the `mark_time` function for better clarity.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.
        """
        return self.mark_time(label, cumulative)

    def end_time(self, label: str, cumulative: bool = False):
        if label not in self.timestamps:
            raise KeyError(f"Label {label} was not found.")
        return self.mark_time(label, cumulative)

    def mark_time(self, label: str, cumulative: bool = False):
        """Given a `label`, add a new timestamp if `label` isn't found, otherwise
        mark the end of the timestamp and add a new duration.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.

        """
        if label not in self.timestamps:
            self.timestamps[label] = [dt.datetime.now(), None]
            self.durations["time_" + label] = 0
        else:
            self.timestamps[label][1] = dt.datetime.now()
            this_segment = self.timestamps[label]
            if cumulative:
                self.durations["time_" + label] += (
                    this_segment[1] - this_segment[0]
                ).total_seconds()
            else:
                self.durations["time_" + label] = (
                    this_segment[1] - this_segment[0]
                ).total_seconds()

    def mark_memory(self, mem_usage: list, label: str):
        """Record the memory usage for a given section of the code.

        Args:
            mem_usage (list): List containing the memory usage and timestamps.
            label (str): One of "fit", "predict", "test".

        Raises:
            KeyError: Raise KeyError if the label is not correct.
        """
        if label in self.memory_usage:
            self.memory_usage[label] = mem_usage
            self.memory_usage[f"peak_{label}"] = max(
                _[0] for _ in self.memory_usage[label]
            )
        else:
            raise KeyError(f"Label {label} not found in mem_usage.")

    def to_dict(self):
        values = [
            self.data_lake_version,
            self.index_name,
            self.index_parameters.get("n_jobs", 1),
            self.query_parameters.get("base_table", ""),
            self.query_parameters.get("query_column", ""),
            self.step,
            self.durations.get("time_create", 0),
            self.durations.get("time_save", 0),
            self.durations.get("time_load", 0),
            self.durations.get("time_query", 0),
            self.memory_usage.get("peak_create", 0),
            self.memory_usage.get("peak_query", 0),
            self.query_results.get("n_candidates", 0),
        ]

        return dict(zip(self.header, values))

    def to_logfile(self):
        if Path(self.log_path).exists():
            with open(self.log_path, "a") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.header)
                writer.writerow(self.to_dict())
        else:
            with open(self.log_path, "w") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.header)
                writer.writeheader()
                writer.writerow(self.to_dict())

    def write_to_json(self, root_path="results/profiling/"):
        res_dict = copy.deepcopy(vars(self))
        res_dict["index_parameters"] = {
            k: str(v) for k, v in res_dict["index_parameters"].items()
        }
        res_dict["timestamps"] = {
            k: v.isoformat()
            for k, v in res_dict["timestamps"].items()
            if isinstance(v, dt.datetime)
        }

        if Path(root_path).exists():
            dest_path = Path(root_path, self.exp_name + ".json")
            with open(dest_path, "w") as fp:
                json.dump(res_dict, fp, indent=2)
        else:
            raise IOError(f"Invalid path {root_path}")
