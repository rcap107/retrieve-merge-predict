import json
import pickle
from operator import itemgetter
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl


def cast_features(table: pl.DataFrame):
    for col in table.columns:
        try:
            table = table.with_columns(
                pl.col(col).cast(pl.Float64)
            )
        except pl.ComputeError:
            continue    
    
    cat_features = [k for k,v in table.schema.items() if str(v) == "Utf8"]
    num_features = [k for k,v in table.schema.items() if str(v) == "Float64"]
    
    return table, num_features, cat_features

class MetadataIndex:
    """This class defines a Metadata Index, which reads the metadata folder to
    have a ready index of the tables found in the data lake. The index is implemented
    as a dictionary where the keys are the hash IDs of the datasets, while the
    metadata of the dataset is the value.
    """

    def __init__(self, metadata_dir=None, index_path=None) -> None:
        """Initialize the class by providing either the metadata directory, or
        the path to a pre-initialized index to load.

        Args:
            metadata_dir (str, optional): Path to the metadata directory. Defaults to None.
            index_path (str, optional): Path to the pre-built index. Defaults to None.

        Raises:
            IOError: Raise IOError if the provided metadata dir does not exist, or isn't a directory.
            ValueError: Raise ValueError if both `metadata_dir` and `index_path` are None.
        """

        if index_path is not None:
            index_path = Path(index_path)
            self.index = self._load_index(index_path)
        elif metadata_dir is not None:
            metadata_dir = Path(metadata_dir)
            if (not metadata_dir.exists()) or (not metadata_dir.is_dir()):
                raise IOError(f"Metadata path invalid.")
            self.index = self.create_index(metadata_dir)
        else:
            raise ValueError("Either `metadata_dir` or `index_path` must be provided.")

    def query_by_hash(self, hashes: Union[list, str]):
        """Given either a hash or a list of hashes, return the metadata of all
        corresponding hashes.

        Args:
            hashes (Union[list, str]): Hash or list of hashes to extract from the index.

        Raises:
            TypeError: Raise TypeError if `hashes` is not a list or a string.

        Returns:
            _type_: The metadata corresponding to the queried hashes.
        """
        # return itemgetter(hashes)(self.index)
        if isinstance(hashes, list):
            return itemgetter(*hashes)(self.index)
        elif isinstance(hashes, str):
            return itemgetter(hashes)(self.index)
        else:
            raise TypeError("Inappropriate type passed to argument `hashes`")
        
    def create_index(self, metadata_dir):
        """Fill the index dictionary by loading all json files found in the provided directory. The stem of the file name will be used as dictionary key.

        Args:
            metadata_dir (Path): Path to the metadata directory.

        Raises:
            IOError: Raise IOError if the metadata directory is invalid.

        Returns:
            dict: Dictionary containing the metadata, indexed by file hash.
        """
        index = {}
        if metadata_dir.exists() and metadata_dir.is_dir:
            all_files = metadata_dir.glob("**/*.json")
            for fpath in all_files:
                md_hash = fpath.stem
                with open(fpath, "r") as fp:
                    index[md_hash] = json.load(fp)
            return index
        else:
            raise IOError(f"Incorrect path {metadata_dir}")

    def _load_index(self, index_path: Path):
        """Load a pre-built index from the given `index_path`.

        Args:
            index_path (Path): Path to the pre-built index.

        Raises:
            IOError: Raise IOError if the index_path is invalid.

        Returns:
            dict: Dictionary containing the metadata.
        """
        if Path(index_path).exists():
            with open(index_path, "rb") as fp:
                index = pickle.load(fp)
            return index
        else:
            raise IOError(f"Index file {index_path} does not exist.")

    def save_index(self, output_path):
        """Save the index dictionary in the given `output_path` for persistence.

        Args:
            output_path (Path): Output path to use when saving the file.
        """
        with open(output_path, "wb") as fp:
            pickle.dump(self.index, fp)

    def fetch_metadata_by_hash(self, tgt_hash):
        try:
            return self.index[tgt_hash]
        except KeyError:
            raise KeyError(f"Hash {tgt_hash} not found in the index.")
