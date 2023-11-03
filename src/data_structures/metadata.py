import hashlib
import json
import os
import pickle
from operator import itemgetter
from pathlib import Path
from typing import Union

import polars as pl

from src.data_structures.indices import LazoIndex, MinHashIndex

QUERY_RESULTS_PATH = Path("results/query_results")
os.makedirs(QUERY_RESULTS_PATH, exist_ok=True)


class MetadataIndex:
    """This class defines a Metadata Index, which reads the metadata folder to
    have a ready index of the tables found in the data lake. The index is implemented
    as a dictionary where the keys are the hash IDs of the datasets, while the
    metadata of the dataset is the value.
    """

    def __init__(
        self, data_lake_variant=None, metadata_dir=None, index_path=None
    ) -> None:
        """Initialize the class by providing either the metadata directory, or
        the path to a pre-initialized index to load.

        Args:
            metadata_dir (str, optional): Path to the metadata directory. Defaults to None.
            index_path (str, optional): Path to the pre-built index. Defaults to None.

        Raises:
            IOError: Raise IOError if the provided metadata dir does not exist, or isn't a directory.
            ValueError: Raise ValueError if both `metadata_dir` and `index_path` are None.
        """
        self.data_lake_variant = data_lake_variant
        if index_path is not None:
            index_path = Path(index_path)
            self.index = self._load_index(index_path)
        elif metadata_dir is not None:
            metadata_dir = Path(metadata_dir)
            if (not metadata_dir.exists()) or (not metadata_dir.is_dir()):
                raise IOError("Metadata path invalid.")
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

    def get_chosen_hash(self, case):
        if case == "source":
            return self.source_metadata["hash"]
        elif case == "candidate":
            return self.candidate_metadata["hash"]
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

    def get_joinpath_str(self, sep=","):
        if len(self.left_on) == 1:
            jps = sep.join(
                [
                    self.source_metadata["full_path"],
                    self.source_metadata["df_name"],
                    self.left_on[0],
                    self.candidate_metadata["full_path"],
                    self.candidate_metadata["df_name"],
                    self.right_on[0],
                ]
            )
            return jps
        else:
            raise NotImplementedError


class RawDataset:
    def __init__(self, full_df_path, source_dl, metadata_dir) -> None:
        self.path = Path(full_df_path)

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

    def __getitem__(self, item):
        return self.metadata[item]

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


class QueryResult:
    def __init__(
        self,
        index: MinHashIndex | LazoIndex,
        source_mdata: RawDataset,
        query_column: str,
        mdata_index: MetadataIndex,
    ) -> None:

        self.index_name = index.index_name
        self.data_lake_version = mdata_index.data_lake_variant
        self.source_mdata = source_mdata
        self.query_column = query_column
        self.candidates = {}

        df = pl.read_parquet(self.source_mdata.path)

        if self.query_column not in df.columns:
            raise pl.ColumnNotFoundError()
        query = df[self.query_column].drop_nulls()

        query_result = index.query_index(query)

        tmp_cand = {}
        for res in query_result:
            hash_, column, similarity = res
            mdata_cand = mdata_index.query_by_hash(hash_)
            cjoin = CandidateJoin(
                indexing_method=index.index_name,
                source_table_metadata=self.source_mdata.metadata,
                candidate_table_metadata=mdata_cand,
                how="left",
                left_on=self.query_column,
                right_on=column,
                similarity_score=similarity,
            )
            tmp_cand[cjoin.candidate_id] = cjoin

        ranked_results = dict(
            sorted(
                [(k, v.similarity_score) for k, v in tmp_cand.items()],
                key=lambda x: x[1],
                reverse=True,
            )
        )

        self.candidates = {k: tmp_cand[k] for k in ranked_results}

    def select_top_k(self, top_k):
        if top_k > 0:
            self.candidates = {
                k: v
                for idx, (k, v) in enumerate(self.candidates.items())
                if idx < top_k
            }

    def save_to_pickle(self):
        output_name = "{}__{}__{}__{}.pickle".format(
            self.data_lake_version,
            self.index_name,
            self.source_mdata.df_name,
            self.query_column,
        )

        with open(Path(QUERY_RESULTS_PATH, output_name), "wb") as fp:
            pickle.dump(self, fp)
