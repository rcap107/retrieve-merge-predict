import hashlib
import json
import os
import pickle
import sqlite3
from operator import itemgetter
from pathlib import Path
from typing import Union

import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data_structures.join_discovery_methods import LazoIndex, MinHashIndex
from src.data_structures.metadata import RawDataset

QUERY_RESULTS_PATH = Path("results/query_results")
os.makedirs(QUERY_RESULTS_PATH, exist_ok=True)


class MetadataIndex:
    """This class defines a Metadata Index, which reads the metadata folder to
    have a ready index of the tables found in the data lake. The index is implemented
    as a dictionary where the keys are the hash IDs of the datasets, while the
    metadata of the dataset is the value.
    """

    def __init__(
        self, db_path="data/metadata/metadata.db", data_lake_path=None, n_jobs=-1
    ) -> None:
        self.db_path = db_path
        self.data_lake_path = Path(data_lake_path)
        self.data_lake_variant = self.data_lake_path.stem
        self.n_jobs = n_jobs

    def build_index(self):
        # logger.info("Case %s", case)
        data_folder = Path(data_folder)
        case = data_folder.stem

        total_files = sum(1 for f in data_folder.glob("**/*.parquet"))

        r = Parallel(n_jobs=1, verbose=0)(
            delayed(self.insert_single_table)(dataset_path, self.data_lake_variant)
            for dataset_path in tqdm(
                data_folder.glob("**/*.parquet"), total=total_files
            )
        )

        return r

    def insert_single_table(self, dataset_path, dataset_source):
        ds = RawDataset(dataset_path, dataset_source)
        return ds.get_as_tuple()

    def query_by_hash(self, hashes: Union[list, str]):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            q_query_by_hash = f"SELECT * FROM binary_update WHERE hash IN ({','.join(['?'] * len(q_list))})"
            res = cur.execute(q_query_by_hash, hashes)
            fetch = res.fetchall()
            con.commit()
        return fetch

    def create_index(self, drop_if_exists=True):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            if drop_if_exists:
                cur.execute(f"DROP TABLE IF EXISTS {self.data_lake_variant};")
            q_create_table = f"CREATE TABLE {self.data_lake_variant}(hash TEXT PRIMARY KEY, table_name TEXT, table_full_path TEXT)"
            cur.execute(q_create_table)
            con.commit()

    def save_index(self, output_path):
        pass

    def fetch_metadata_by_hash(self, tgt_hash):
        pass


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
    def __init__(self, full_df_path, source_dl) -> None:
        self.path = Path(full_df_path)

        if not self.path.exists():
            raise IOError(f"File {self.path} not found.")

        # self.df = self.read_dataset_file()
        self.hash = self.prepare_path_digest()
        self.df_name = self.path.stem
        self.source_dl = source_dl

        self.metadata = {
            "full_path": str(self.path),
            "hash": self.hash,
            "df_name": self.df_name,
            "source_dl": source_dl,
            "license": "",
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

    def get_as_tuple(self):
        return (self.hash, self.df_name, str(self.path))

    # def save_metadata_to_db(self, cursor: sqlite3.Cursor):
    # query = f"{self.df_name}, "

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
        self.n_candidates = len(self.candidates)

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
