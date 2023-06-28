import logging
from pathlib import Path
import polars as pl
import datetime as dt

from time import process_time

RUN_ID_PATH = Path("results/run_id")
SCENARIO_ID_PATH = Path("results/scenario_id")

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="a",
    filename="data/logging.txt",
    level=logging.DEBUG,
)


class ScenarioLogger:
    def __init__(
        self,
        source_table,
        git_hash,
        iterations,
        join_strategy,
        aggregation,
        target_dl,
        n_splits,
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
        self.scenario_id = self.find_latest_scenario_id()
        self.run_id = 0
        self.start_timestamp = None
        self.end_timestamp = None
        self.source_table = source_table
        self.git_hash = git_hash
        self.iterations = iterations
        self.join_strategy = join_strategy
        if join_strategy == "nojoin":
            self.aggregation = "nojoin"
        else:
            self.aggregation = aggregation
        self.target_dl = target_dl
        self.n_splits = n_splits
        self.results = {}
        self.process_time = 0

    def add_timestamp(self, which_ts):
        self.timestamps[which_ts] = dt.datetime.now()

    def add_process_time(self):
        self.process_time = process_time()

    def find_latest_scenario_id(self):
        """Utility function for opening the scenario_id file, checking for errors and
        incrementing it by one at the start of a run.

        Raises:
            ValueError: Raise ValueError if the read scenario_id is not a positive integer.

        Returns:
            int: The new (incremented) scenario_id.
        """
        if SCENARIO_ID_PATH.exists():
            with open(SCENARIO_ID_PATH, "r") as fp:
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
            with open(SCENARIO_ID_PATH, "w") as fp:
                fp.write(f"{scenario_id}")
        else:
            scenario_id = 0
            with open(SCENARIO_ID_PATH, "w") as fp:
                fp.write(f"{scenario_id}")
        return scenario_id

    def get_next_run_id(self):
        self.run_id += 1
        next_run_id = self.run_id
        return next_run_id

    def to_string(self):
        str_res = ",".join(
            map(
                str,
                [
                    self.scenario_id,
                    self.git_hash,
                    self.source_table,
                    self.iterations,
                    self.join_strategy,
                    self.aggregation,
                    self.target_dl,
                    self.n_splits,
                    self.results["n_candidates"],
                ],
            )
        )
        str_res += ","
        for ts in self.timestamps.values():
            str_res += str(ts) + ","

        return str_res.rstrip(",")

    def pretty_print(self):
        print(f"Scenario ID: {self.scenario_id}")
        print(f"Source table: {self.source_table}")
        print(f"Iterations: {self.iterations}")
        print(f"Join strategy: {self.join_strategy}")
        print(f"Aggregation: {self.aggregation}")
        print(f"DL Variant: {self.target_dl}")

    def write_to_file(self, out_path):
        with open(out_path, "a") as fp:
            fp.write(self.to_string() + "\n")


class RunLogger:
    def __init__(self, scenario_logger: ScenarioLogger, fold_id, additional_parameters):
        # TODO: rewrite with __getitem__ instead
        self.scenario_id = scenario_logger.scenario_id
        self.fold_id = fold_id
        self.run_id = scenario_logger.get_next_run_id()
        self.status = None
        self.timestamps = {}
        self.durations = {}
        self.parameters = self.get_parameters(scenario_logger, additional_parameters)
        self.results = {}

        self.mark_time("run")

    def get_parameters(self, scenario_logger: ScenarioLogger, additional_parameters):
        parameters = {
            "source_table": scenario_logger.source_table,
            "candidate_table": "",
            "left_on": "",
            "right_on": "",
            "git_hash": scenario_logger.git_hash,
            "index_name": "base_table",
            "iterations": scenario_logger.iterations,
            "join_strategy": scenario_logger.join_strategy,
            "aggregation": scenario_logger.aggregation,
            "target_dl": scenario_logger.target_dl,
        }
        if additional_parameters is not None:
            parameters.update(additional_parameters)

        return parameters

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
        return self.timestamps[label]

    def to_str(self):
        res_str = ",".join(
            map(
                str,
                [
                    self.scenario_id,
                    self.run_id,
                    self.status,
                    self.parameters["target_dl"],
                    self.parameters["git_hash"],
                    self.parameters["index_name"],
                    self.parameters["source_table"],
                    self.parameters["candidate_table"],
                    self.parameters["iterations"],
                    self.parameters["join_strategy"],
                    self.parameters["aggregation"],
                    self.fold_id,
                    self.durations["time_train"],
                    self.durations["time_eval"],
                    self.durations.get("time_join", ""),
                    self.durations.get("time_eval_join", ""),
                    self.results.get("rmse", ""),
                    self.results.get("r2score", ""),
                ],
            )
        )
        return res_str
