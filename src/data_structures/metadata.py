import hashlib
import json
import pickle
from operator import itemgetter
from pathlib import Path
from typing import Union

import polars as pl


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

    def get_join_information(self):
        return (
            self.source_metadata,
            self.candidate_metadata,
            self.left_on,
            self.right_on,
        )


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

        self.metadata = {
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
            json.dump(self.metadata, fp, indent=2)

    def prepare_metadata(self):
        pass

    def save_to_json(self):
        pass

    def save_to_csv(self):
        pass

    def save_to_parquet(self):
        pass
