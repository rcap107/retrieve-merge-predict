import copy
import csv
import datetime as dt
import json
import logging
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

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
        base_table,
        git_hash,
        iterations,
        chosen_model,
        aggregation,
        target_dl,
        n_splits,
        top_k,
        jd_method="minhash",
        task="regression",
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
        self.task = task
        self.run_id = 0
        self.start_timestamp = None
        self.end_timestamp = None
        self.chosen_model = chosen_model
        self.jd_method = jd_method
        self.base_table = base_table
        self.git_hash = git_hash
        self.iterations = iterations
        self.aggregation = aggregation
        self.target_dl = target_dl
        self.n_splits = n_splits
        self.top_k = top_k
        self.results = None
        self.process_time = 0
        self.status = None
        self.exception_name = None
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
            "base_table": self.base_table,
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
                    self.base_table,
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

    def pretty_print(self):
        print(f"Run name: {self.exp_name}")
        print(f"Scenario ID: {self.scenario_id}")
        print(f"Base table: {self.base_table}")
        print(f"Iterations: {self.iterations}")
        print(f"Aggregation: {self.aggregation}")
        print(f"DL Variant: {self.target_dl}")

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
            k: v.isoformat() for k, v in res_dict["timestamps"].items()
        }
        if Path(root_path).exists():
            with open(
                Path(root_path, self.exp_name, "json", f"{self.scenario_id}.json"), "w"
            ) as fp:
                json.dump(res_dict, fp, indent=2)
        else:
            raise IOError(f"Invalid path {root_path}")

    def write_summary_plot(self, root_path):
        keys = [
            "exp_name",
            "scenario_id",
            "chosen_model",
            "base_table",
            "git_hash",
            "iterations",
            "target_dl",
            "aggregation",
        ]
        dict_print = {k: v for k, v in vars(self).items() if k in keys}
        annotation = json.dumps(dict_print, indent=2)
        fig = plt.figure()
        axs = fig.subplots(nrows=2)
        sns.boxplot(
            data=self.results.to_pandas(), y="estimator", x="r2", ax=axs[0], orient="h"
        )
        axs[1].text(0, 0, annotation)
        axs[1].axis("off")
        plt.tight_layout()
        fig_path = Path(root_path, self.exp_name, "plots", f"{self.scenario_id}.png")
        fig.savefig(fig_path)

    def finish_run(self, root_path="results/logs/"):
        if not self.debug:
            self.write_summary_plot(root_path)
            self.write_to_json(root_path)
        else:
            print("ScenarioLogger is in debugging mode. No files will be created.")


class RunLogger:
    def __init__(
        self,
        scenario_logger: ScenarioLogger,
        additional_parameters: dict,
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
        self.results = {}
        self.additional_info = {
            "best_candidate_hash": None,
            "left_on": None,
            "right_on": None,
            "joined_columns": None,
        }

        self.mark_time("run")

    def get_parameters(self, scenario_logger: ScenarioLogger, additional_parameters):
        parameters = {
            "base_table": scenario_logger.base_table,
            "candidate_table": "",
            "left_on": "",
            "right_on": "",
            "git_hash": scenario_logger.git_hash,
            "iterations": scenario_logger.iterations,
            "aggregation": scenario_logger.aggregation,
            "target_dl": scenario_logger.target_dl,
            "fold_id": "",
        }
        if additional_parameters is not None:
            parameters.update(additional_parameters)

        return parameters

    def update_parameters(self, additional_parameters):
        self.parameters.update(additional_parameters)

    def update_timestamps(self, additional_timestamps=None):
        if additional_timestamps is not None:
            self.timestamps.update(additional_timestamps)

    def set_run_status(self, status):
        """Set run status for logging.

        Args:
            status (str): Status to use.
        """
        self.status = status

    def start_time(self, label, cumulative=False):
        """Wrapper around the `mark_time` function for better clarity.

        Args:
            label (str): Label of the operation to mark.
            cumulative (bool, optional): If set to true, all operations performed with the same label
            will add up to a total duration rather than being marked independently. Defaults to False.
        """
        return self.mark_time(label, cumulative)

    def end_time(self, label, cumulative=False):
        if label not in self.timestamps:
            raise KeyError(f"Label {label} was not found.")
        return self.mark_time(label, cumulative)

    def mark_time(self, label, cumulative=False):
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

    def get_time(self, label):
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

    def get_duration(self, label):
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

    def measure_results(self, y_true, y_pred):
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("The provided vectors have inconsistent shapes.")

        if self.task == "regression":
            self.results["r2"] = r2_score(y_true, y_pred)
            self.results["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
            return self.results
        elif self.task == "classification":
            raise NotImplementedError
        else:
            raise ValueError()

    def set_additional_info(self, info_dict: dict):
        self.additional_info.update(info_dict)

    def to_dict(self):
        values = [
            self.scenario_id,
            self.status,
            self.parameters["target_dl"],
            self.parameters["base_table"],
            self.parameters["estimator"],
            self.parameters["aggregation"],
            self.parameters["chosen_model"],
            self.parameters.get("fold_id", ""),
            self.durations.get("time_fit", ""),
            self.durations.get("time_predict", ""),
            self.durations.get("time_run", ""),
            self.results.get("r2", ""),
            self.results.get("rmse", ""),
            self.additional_info.get("best_candidate_hash", ""),
            self.additional_info.get("joined_columns", ""),
            self.additional_info.get("left_on", ""),
            self.additional_info.get("right_on", ""),
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


class RawLogger(RunLogger):
    def __init__(
        self,
        scenario_logger: ScenarioLogger,
        fold_id: int,
        additional_parameters: dict,
        json_path="results/json",
    ):
        super().__init__(scenario_logger, additional_parameters, json_path)
        self.fold_id = fold_id
        self.parameters["fold_id"] = fold_id
