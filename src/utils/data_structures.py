import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Union
import hashlib
import polars as pl
import pandas as pd
import zlib
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


class FakeFileHasher:
    # by https://github.com/ogrisel
    def __init__(self):
        self._hash = hashlib.sha256()

    def read(self, *ignored, **ignored_again):
        raise RuntimeError(f"{self.__class__.__name__} does not support reading.")

    def seek(self, *ignored):
        raise RuntimeError(f"{self.__class__.__name__} does not support seeking.")

    def write(self, some_bytes):
        self._hash.update(some_bytes)
        return len(some_bytes)

    def flush(self):
        pass

    def hexdigest(self):
        return self._hash.hexdigest()


def read_dataset_paths(dataset_list_path: Path):
    valid_paths = []
    with open(dataset_list_path, "r") as fp:
        n_paths = int(fp.readline().strip())
        for idx, row in enumerate(fp):
            valid_paths.append(Path(row.strip()))
    stems = [pth.stem for pth in valid_paths]
    return valid_paths, stems


class RawDataset:
    def __init__(self, full_df_path, source_dl, metadata_dir) -> None:
        self.path = Path(full_df_path).resolve()

        if not self.path.exists():
            raise IOError(f"File {self.path} not found.")

        # self.df = self.read_dataset_file()
        self.hash = self.prepare_path_digest()
        self.df_name = self.path.stem
        self.source_dl = source_dl
        self.path_metadata = Path(metadata_dir, self.hash + ".json")

        self.info = {
            "full_path": str(self.path),
            "hash": self.hash,
            "df_name": self.df_name,
            "source_dl": source_dl,
            "license": "",
            "path_metadata": str(self.path_metadata.resolve()),
        }

    def read_dataset_file(self):
        if self.path.suffix == ".csv":
            # TODO Add parameters for the `pl.read_csv` function
            return pl.read_csv(self.path)
        elif self.path.suffix == ".parquet":
            # TODO Add parameters for the `pl.read_parquet` function
            return pl.read_parquet(self.path)
        else:
            raise IOError(f"Extension {self.path.suffix} not supported.")

    def prepare_path_digest(self):
        hash_ = hashlib.md5()
        hash_.update(str(self.path).encode())
        return hash_.hexdigest()

    def save_metadata_to_json(self, metadata_dir=None):
        if metadata_dir is None:
            pth_md = self.path_metadata
        else:
            pth_md = Path(metadata_dir, self.hash + ".json")
        with open(pth_md, "w") as fp:
            json.dump(self.info, fp, indent=2)

    def prepare_metadata(self):
        pass

    def save_to_json(self):
        pass

    def save_to_csv(self):
        pass

    def save_to_parquet(self):
        pass


class Dataset:
    def __init__(self, path_metadata) -> None:
        self.path_metadata = Path(path_metadata)
        self.path = Path(df_path)

        if not self.path.exists():
            raise IOError(f"File {self.path} not found.")

        self.id = df_name
        self.table = self.read_dataset_file()
        self.source_df = source_dl
        self.source_og = source_og
        self.dataset_license = dataset_license
        self.path_metadata = None

        self.md5hash = None

    def read_metadata(self):
        if self.path_metadata.exists():
            self.metadata_dict = json.load(self.path_metadata)
        else:
            raise IOError(f"File {self.path_metadata} not found.")

    def read_dataset_file(self):
        if self.path.suffix == ".csv":
            # TODO Add parameters for the `pl.read_csv` function
            return pl.read_csv(self.path)
        elif self.path.suffix == ".parquet":
            # TODO Add parameters for the `pl.read_parquet` function
            return pl.read_parquet(self.path)
        else:
            raise IOError(f"Extension {self.path.suffix} not supported.")

    def prepare_hash(self, fp, block_size=2**20):
        md5 = hashlib.md5()
        while True:
            data = fp.read(block_size)
            if not data:
                break
            md5.update(data)
        return md5.hexdigest()

    def prepare_dataset_digest(self):
        ffh = FakeFileHasher()
        self.df.write_ipc(ffh)
        return ffh.hexdigest()

    def save_metadata_to_json(self):
        pass

    def prepare_metadata(self):
        pass

    def save_to_csv(self):
        pass

    def save_to_parquet(self):
        pass


class CandidateDataset(Dataset):
    def __init__(
        self, df_name, df_path, source_dl, source_og=None, dataset_license=None
    ) -> None:
        super().__init__(df_name, df_path, source_dl, source_og, dataset_license)
        self.candidate_for = None


class SourceDataset(Dataset):
    def __init__(
        self,
        df_name,
        df_path,
        source_dl,
        source_og=None,
        dataset_license=None,
        task=None,
        target_column=None,
        metric=None,
    ) -> None:
        super().__init__(df_name, df_path, source_dl, source_og, dataset_license)
        self.task = task
        self.target_column = target_column
        self.metric = metric

        self.candidates = {}

        # TODO: add checks for the args above

    def add_candidate_table(
        self,
        candidate_dataset: CandidateDataset,
        rel_type,
        left_on=None,
        right_on=None,
        similarity_score=None,
    ):
        # TODO: add check for type
        if candidate_dataset.id not in self.candidates:
            self.candidates[candidate_dataset.id] = []
            new_cand_rel = CandidateRelationship(
                self.id,
                candidate_dataset.id,
                rel_type=rel_type,
                left_on=left_on,
                right_on=right_on,
                similarity_score=similarity_score,
            )
            self.candidates[candidate_dataset.id].append(new_cand_rel)


class IntegratedDataset(Dataset):
    def __init__(
        self,
        df_name,
        df_path,
        source_dl,
        source_id,
        augmentation_list,
        source_og=None,
        dataset_license=None,
    ) -> None:
        super().__init__(df_name, df_path, source_dl, source_og, dataset_license)
        self.source_id = source_id
        self.augmentation_list = augmentation_list


class CandidateJoin:
    def __init__(
        self,
        indexing_method,
        source_table_metadata,
        candidate_table_metadata,
        how=None,
        left_on=None,
        right_on=None,
        on=None,
        similarity_score=None,
    ) -> None:
        self.indexing_method = indexing_method
        self.source_table = source_table_metadata["hash"]
        self.candidate_table = candidate_table_metadata["hash"]
        self.source_metadata = source_table_metadata
        self.candidate_metadata = candidate_table_metadata

        self.similarity_score = similarity_score

        if how not in ["left", "right", "inner", "outer"]:
            raise ValueError(f"Join strategy {how} not recognized.")
        self.how = how

        self.left_on = self._convert_to_list(left_on)
        self.right_on = self._convert_to_list(right_on)
        self.on = self._convert_to_list(on)

        if self.on is not None and all([self.left_on is None, self.right_on is None]):
            self.left_on = self.right_on = [self.on]

        self.candidate_id = self.generate_candidate_id()

    @staticmethod
    def _convert_to_list(val):
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            return [val]
        elif val is None:
            return None
        else:
            raise TypeError

    def get_chosen_path(self, case):
        if case == "source":
            return self.source_metadata["full_path"]
        elif case == "candidate":
            return self.candidate_metadata["full_path"]
        else:
            raise ValueError

    def generate_candidate_id(self):
        """Generate a unique id for this candidate relationship. The same pair of tables can have multiple candidate
        relationships, so this function takes the index, source table, candidate table, left/right columns and combines them
        to produce a unique id.
        """
        join_string = [
            self.indexing_method,
            self.source_table,
            self.candidate_table,
            self.how + "_j",
        ]

        if self.left_on is not None and self.right_on is not None:
            join_string += ["_".join(self.left_on)]
            join_string += ["_".join(self.right_on)]
        elif self.on is not None:
            join_string += ["_".join(self.on)]

        id_str = "_".join(join_string).encode()

        md5 = hashlib.md5()
        md5.update(id_str)
        return md5.hexdigest()


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
        str_res += ','
        for ts in self.timestamps.values():
            str_res += str(ts) + ","
        
        return str_res.rstrip(",")

    def write_to_file(self, out_path):
        with open(out_path, "a") as fp:
            fp.write(self.to_string() + "\n")


class RunLogger:
    def __init__(self, scenario_logger: ScenarioLogger, fold_id, additional_parameters):
        # TODO: rewrite with __getitem__ instead
        self.scenario_id = scenario_logger.scenario_id
        self.fold_id = fold_id
        self.run_id = self.find_latest_run_id()
        self.status = None
        self.timestamps = {}
        self.durations = {}
        self.parameters = self.get_parameters(scenario_logger, additional_parameters)
        self.results = {}

        self.add_time("run_start_time")

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
        self.status = status

    def find_latest_run_id(self):
        """Utility function for opening the run_id file, checking for errors and
        incrementing it by one at the start of a run.

        Raises:
            ValueError: Raise ValueError if the read run_id is not a positive integer.

        Returns:
            int: The new (incremented) run_id.
        """
        if RUN_ID_PATH.exists():
            with open(RUN_ID_PATH, "r") as fp:
                last_run_id = fp.read().strip()
                try:
                    run_id = int(last_run_id) + 1
                except ValueError:
                    raise ValueError(
                        f"Run ID {last_run_id} is not a positive integer. "
                    )
                if run_id < 0:
                    raise ValueError(f"Run ID {run_id} is not a positive integer. ")
            with open(RUN_ID_PATH, "w") as fp:
                fp.write(f"{run_id}")
        else:
            run_id = 0
            with open(RUN_ID_PATH, "w") as fp:
                fp.write(f"{run_id}")
        return run_id

    def add_time(self, label, value=None):
        """Add a new timestamp starting __now__, with the given label.

        Args:
            label (str): Label to assign to the timestamp.
        """
        if value is None:
            self.timestamps[label] = dt.datetime.now()
        else:
            self.timestamps[label] = -1

    def get_time(self, label):
        """Retrieve a time according to the given label.

        Args:
            label (str): Label of the timestamp to be retrieved.
        Returns:
            _type_: Retrieved timestamp.
        """
        return self.timestamps[label]

    def add_duration(self, label_start=None, label_end=None, label_duration=None):
        if label_start is None and label_end is None:
            if label_duration is not None:
                self.durations[label_duration] = -1
            else:
                raise ValueError(f"`label_duration` is required.")
        else:
            assert label_start in self.timestamps
            assert label_end in self.timestamps

            self.durations[label_duration] = (
                self.timestamps[label_end] - self.timestamps[label_start]
            ).total_seconds()

    def to_str(self):
        res_str = ",".join(
            map(
                str,
                [
                    self.scenario_id,
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
                    # self.durations["fit_time"],
                    # self.durations["score_time"],
                    # self.timestamps.get("join_start", ""),
                    # self.timestamps.get("join_end", ""),
                    # self.durations.get("join_duration", ""),
                    # self.parameters.get("similarity", ""),
                    # self.parameters.get("size_prejoin", ""),
                    # self.parameters.get("size_postjoin", ""),
                    self.results.get("rmse", ""),
                    self.results.get("r2score", ""),
                    
                ],
            )
        )
        return res_str
