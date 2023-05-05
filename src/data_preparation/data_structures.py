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


logging.basicConfig(
    format="%(asctime)s %(message)s",
    filemode="a",
    filename="data/logging.txt",
    level=logging.DEBUG,
)


def read_dataset_paths(dataset_list_path: Path):
    valid_paths = []
    with open(dataset_list_path, "r") as fp:
        n_paths = int(fp.readline().strip())
        for idx, row in enumerate(fp):
            valid_paths.append(Path(row.strip()))
    stems = [pth.stem for pth in valid_paths]
    return valid_paths, stems


class Dataset:
    def __init__(
        self, df_name, df_path, source_dl, source_og=None, dataset_license=None
    ) -> None:
        self.path = Path(df_path)
        
        if not self.path.exists():
            raise IOError(f"File {self.path} not found.")
        
        self.id = df_name
        self.table = self.read_dataset_file()
        self.source_df = source_dl
        self.source_og = source_og
        self.license = dataset_license
        self.path_metadata = None

        self.md5hash = None

    def read_dataset_file(self):
        if self.path.suffix == ".csv":
            #TODO Add parameters for the `pl.read_csv` function
            return pl.read_csv(self.path)
        elif self.path.suffix == ".parquet":
            #TODO Add parameters for the `pl.read_parquet` function
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

    def save_metadata_to_json(self):
        pass

    def prepare_metadata(self):
        pass

    def save_to_json(self):
        json.dump(
            self.to_dict(),
            indent=2,
        )
        
    def save_to_csv(self):
        pass
    
    def save_to_parquet(self):
        pass


class CandidateDataset(Dataset):
    def __init__(
        self, df_name, df_path, source_dl, source_og=None, license=None
    ) -> None:
        super().__init__(df_name, df_path, source_dl, source_og, license)
        self.candidate_for = None


class SourceDataset(Dataset):
    def __init__(
        self,
        df_name,
        df_path,
        source_dl,
        source_og=None,
        license=None,
        task=None,
        target_column=None,
        metric=None,
    ) -> None:
        super().__init__(df_name, df_path, source_dl, source_og, license)
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
    def __init__(self, df_name, df_path, source_dl, source_id, augmentation_list, source_og=None, license=None) -> None:
        super().__init__(df_name, df_path, source_dl, source_og, license)
        self.source_id = source_id
        self.augmentation_list = augmentation_list
        
class CandidateRelationship:
    def __init__(
        self,
        source_table,
        candidate_table,
        rel_type=None,
        left_on=None,
        right_on=None,
        similarity_score=None,
    ) -> None:
        self.source_id = source_table
        self.candidate_table = candidate_table
        self.similarity_score = similarity_score

        self.rel_type = rel_type
        if rel_type not in ["join", "union"]:
            raise ValueError(f"Relation type {rel_type} not recognized.")
        else:
            self.left_on, self.right_on = left_on, right_on

        self.candidate_id = self.generate_candidate_id()

    def generate_candidate_id(self):
        """Generate a unique id for this candidate relationship. The same pair of tables can have multiple candidate
        relationships, so this function takes the source table, candidate table, left/right columns and combines them
        to produce a unique id.
        """
        id_str = "_".join(
            [
                self.source_id,
                self.candidate_table,
                self.rel_type,
                self.left_on,
                self.right_on,
            ]
        ).encode()
        md5 = hashlib.md5()
        md5.update(id_str)
        return md5.hexdigest()
