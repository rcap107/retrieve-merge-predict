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

RUN_ID_PATH=Path("results/run_id")

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


class RunResult:
    def __init__(
        self,
    ):
        self.run_id = self.find_latest_run_id()
        self.obj = {}
        self.status = None
        self.obj["run_id"] = self.run_id
        self.obj["status"] = None
        self.obj["timestamps"] = {}
        self.obj["durations"] = {}
        self.obj["parameters"] = {}
        # Losses and actual imputation results
        self.obj["results"] = {}
        # Statistics measured on the given dataset (% missing values etc)
        self.obj["statistics"] = {}

        self.add_time("run_start_time")

    
    def set_run_status(self, status):
        self.obj["status"] = status
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

    def add_value(self, obj_name, key, value):
        """Updating a single value in a given object. 

        Args:
            obj_name (str): Label of the object to update.
            key (_type_): Key of the object to update.
            value (_type_): Value of the object to update.
        """
        self.obj[obj_name][key] = value

    def get_value(self, obj_name, key):
        """Retrieve a single value from one of the dictionaries. 

        Args:
            obj_name (str): Label of the object to query.
            key (_type_): Dict key to use to retrieve the value.

        Returns:
            _type_: Retrieved value.
        """
        return self.obj[obj_name][key]

    def add_time(self, label, value=None):
        """Add a new timestamp starting __now__, with the given label. 

        Args:
            label (str): Label to assign to the timestamp.
        """
        if value is None:
            self.obj["timestamps"][label] = dt.datetime.now()
        else:
            self.obj["timestamps"][label] = -1
        
    def get_time(self, label):
        """Retrieve a time according to the given label.    

        Args:
            label (str): Label of the timestamp to be retrieved. 
        Returns:
            _type_: Retrieved timestamp.
        """
        return self.obj["timestamp"][label]

    def add_duration(self, label_start=None, label_end=None, label_duration=None):
        #TODO: Fix docstring
        if label_start is None and label_end is None:
            if label_duration is not None:
                self.obj["durations"][label_duration] = -1
            else:
                raise ValueError(f"`label_duration` is required.")
        else:
            assert label_start in self.obj["timestamps"]
            assert label_end in self.obj["timestamps"]
            
            self.obj["durations"][label_duration] = (
                self.obj["timestamps"][label_end] - self.obj["timestamps"][label_start]
            ).total_seconds()

    def __getitem__(self, item):
        return self.obj[item]

    def to_dict(self):
        output_dict = {}
        for key, value in self.obj.items():
            if isinstance(value, dict):
                output_dict.update(value)
            else:
                output_dict[key] = value
        return output_dict
