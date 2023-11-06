import json
import logging
import pickle
from pathlib import Path
from sys import getsizeof

import lazo_index_service
import numpy as np
import polars as pl
import polars.selectors as cs
from datasketch import LeanMinHash, MinHash, MinHashLSHEnsemble
from joblib import Parallel, delayed, dump, load
from lazo_index_service.errors import LazoError
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

jd_logger = logging.getLogger("metadata_logger")
jd_logger.setLevel(logging.DEBUG)

sh_logger = logging.getLogger("sh_logger")
sh_logger.setLevel(logging.ERROR)

log_format = "%(message)s"
formatter = logging.Formatter(fmt=log_format)

fh = logging.FileHandler(filename="results/logging_jd.log")
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)

jd_logger.addHandler(fh)
sh_logger.addHandler(sh)

LAZO_MESSAGE_SIZE_LIMIT = 4194304


class BaseIndex:
    def _index_single_table(self):
        pass

    def add_tables_from_dict(self, df_dict):
        pass

    def query_index(self, query):
        pass

    def load_index(self):
        pass

    def save_index(self, output_path):
        pass


class MinHashIndex:
    """
    Index class based on `MinHashLSHEnsemble`. By default, it scans for metadata files
    in the provided `data_dir` and adds all them to the index.

    Since by default the LSHEnsemble queries based on a single threshold defined
    at creation time, this index builds accepts a list of thresholds and creates
    an ensemble for each.

    Ensembles do not support online updates, so after loading all tables in the index it is necessary
    to invoke the function `create_ensembles`. Querying without this step will raise an exception.


    """

    def __init__(
        self,
        data_dir=None,
        thresholds=[20],
        num_perm=128,
        num_part=32,
        oneshot=True,
        index_file=None,
        n_jobs=1,
    ) -> None:
        """
        If `index_file` is provided, the data structures required for the index are loaded from the given
        index file.

        If `oneshot` is set to True, the index will be initialized within this function.
        If `oneshot` is set to False, the index creation will not be wrapped up until the user manually
        invokes `create_ensembles`: this allows to update the indices with tables that were not added
        while scanning `data_dir`.

        Args:
            data_dir (str, optional): Path to the dir that contains the metadata of the target tables.
            thresholds (list, optional): List of thresholds to be used by the ensemble. Defaults to [20].
            num_perm (int, optional): Number of hash permutations. Defaults to 128.
            num_part (int, optional): Number of partitions. Defaults to 32.
            oneshot (bool, optional): If False, index will have to be finalized by the user. Defaults to True.
            index_file (str, optional): Path to a file containing a pre-computed index.
        """
        self.index_name = "minhash"

        self.hash_index = []
        self.num_perm = num_perm
        self.num_part = num_part
        self.thresholds = sorted(thresholds)
        self.initialized = False
        self.ensembles = {}
        self.n_jobs = n_jobs

        if index_file is not None:
            self.load_index(index_file)
            self.initialized = True

        elif data_dir is not None:
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                raise IOError("Invalid data directory")

            self.add_tables_from_path(self.data_dir)

            if oneshot:
                # If oneshot, wrap up the generation of the index here. If not, create_ensemble will have to be called later
                self.create_ensembles()
        else:
            # Do nothing, the user will load the data manually.
            pass

    def _index_single_table(self, path) -> dict:
        with open(path, "r") as fp:
            mdata_dict = json.load(fp)
        ds_hash = mdata_dict["hash"]
        # Selecting only string columns
        df = pl.read_parquet(mdata_dict["full_path"]).select(cs.string())

        minhashes = {}
        for col in df.columns:
            key = ds_hash + "__" + col
            m = MinHash(num_perm=self.num_perm)
            uniques = df[col].drop_nulls().unique().cast(str)
            for u in uniques:
                m.update(u.encode("utf8"))
            lean_m = LeanMinHash(seed=m.seed, hashvalues=m.hashvalues)
            minhashes[key] = (lean_m, len(uniques))
        return minhashes

    def add_tables_from_path(self, data_path):
        """Add tables to the index reading metadata from the given `data_path`.
        `data_path` should contain json files that include the metadata of the file.

        Args:
            data_path (Path): Path to the directory to scan for metadata.
        """
        total_files = sum(1 for f in data_path.glob("*.json"))
        if total_files == 0:
            raise RuntimeError("No metadata files were found.")

        paths = list(data_path.glob("*.json"))
        t_list = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._index_single_table)(path)
            for _, path in tqdm(
                enumerate(paths),
                total=total_files,
                desc="Loading metadata in index",
            )
        )

        t_dict = {}
        for _ in t_list:
            t_dict.update(_)
        self.hash_index += [(key, m, setlen) for key, (m, setlen) in t_dict.items()]

    def create_ensembles(self):
        """Utility function to create the ensembles once all tables have been loaded in the index."""
        if not self.initialized:
            for t in self.thresholds:
                ens = MinHashLSHEnsemble(
                    threshold=t / 100, num_perm=self.num_perm, num_part=self.num_part
                )
                ens.index(self.hash_index)
                self.ensembles[t] = ens
            print("Initialization complete. ")
            self.initialized = True
        else:
            print("Ensembles are already initialized. Skipping.")

    @staticmethod
    def prepare_result(query_result, threshold):
        """Given the `query_result` and the `threshold`, reformat the content
        of `query_result` according to the format required for the later steps
        in the pipeline.

        Args:
            query_result (_type_): Result of querying the datasketch ensembles.
            threshold (int): Threshold used for the ensemble.

        Returns:
            dict: A dictionary with keys (table, column) and value `threshold`.
        """
        result_dict = {}
        for result in query_result:
            table, column = result.split("__", maxsplit=1)

            result_dict[(table, column)] = threshold
        return result_dict

    def query_index(self, query, threshold=None, to_dataframe=False):
        """Query the index with a list of values and return a dictionary that contains all columns
        that satisfy the query for each threshold.

        Args:
            query (Iterable): List of values to query for.
            threshold (Iterable): List of thresholds to be used when querying.
            to_dataframe (bool): If True, convert the output into a Polars dataframe.

        Raises:
            RuntimeError: Raise RunTimeError if initialization was not completed.

        Returns:
            dict: Dictionary that contains the query results.
        """
        if self.initialized:
            query = list(set(query))

            m_query = MinHash(num_perm=self.num_perm)
            for q in query:
                m_query.update(q.encode("utf8"))

            # TODO: turn this into a list with format (table, column, threshold)
            query_dict = {}

            if threshold is not None:
                if any([th not in self.ensembles for th in threshold]):
                    raise ValueError(f"Invalid thresholds in the provided list.")
                else:
                    for th in sorted(threshold):
                        ens = self.ensembles[th]
                        res = list(ens.query(m_query, len(query)))
                        query_dict.update(self.prepare_result(res, threshold))
            else:
                for threshold, ens in self.ensembles.items():
                    res = list(ens.query(m_query, len(query)))
                    query_dict.update(self.prepare_result(res, threshold))

            if to_dataframe:
                query_results = pl.from_records(
                    query_dict, schema=["hash", "column", "threshold"]
                )
            else:
                query_results = []
                for k, v in query_dict.items():
                    query_results.append((k[0], k[1], v))

            return query_results
        else:
            raise RuntimeError("Ensembles are not initialized.")

    def save_ensembles(self, output_path):
        """Save the ensembles to file in the given `output_path`.

        Args:
            output_path (str): Output path.
        """
        with open(output_path, "wb") as fp:
            dump(self.ensembles, fp)

    def save_index(self, output_path):
        out_dict = {
            "index_name": self.index_name,
            "hash_index": self.hash_index,
            "num_perm": self.num_perm,
            "num_part": self.num_part,
            "thresholds": self.thresholds,
            "ensembles": self.ensembles,
        }

        with open(output_path, "wb") as fp:
            dump(out_dict, fp)

    def load_index(self, index_file=None, index_dict=None):
        if index_file is not None:
            if Path(index_file).exists():
                with open(index_file, "rb") as fp:
                    index_dict = load(fp)
                    self.hash_index = index_dict["hash_index"]
                    self.num_perm = index_dict["num_perm"]
                    self.num_part = index_dict["num_part"]
                    self.thresholds = index_dict["thresholds"]
                    self.ensembles = index_dict["ensembles"]
                    self.initialized = True
            else:
                raise FileNotFoundError(f"File `{index_file}` not found.")
        elif index_dict is not None:
            self.hash_index = index_dict["hash_index"]
            self.num_perm = index_dict["num_perm"]
            self.num_part = index_dict["num_part"]
            self.thresholds = index_dict["thresholds"]
            self.ensembles = index_dict["ensembles"]
            self.initialized = True
        else:
            raise ValueError("Either `index_file` or `index_dict` must be provided.")


class LazoIndex:
    """This class implements a wrapper around the lazo index client. The lazo
    index server must be already running on the machine.
    """

    def __init__(
        self,
        data_dir=None,
        partition_size=50_000,
        host="localhost",
        port=15449,
        index_file=None,
    ):
        """Initialize the LazoIndex class.

        Args:
            data_dir (str, optional): If provided, create and initialize the index scanning the
            given `data_dir`.
            partition_size (int, optional): Due to how protobuf works, column domains will be
            partitioned in lists of size `partition_size`. Defaults to 50_000.
            host (str, optional): Lazo server host address. Defaults to "localhost".
            port (int, optional): Lazo server port. Defaults to 15449.
            index_file (str, optional): Path to pre-computed index config.
        """
        self.index_name = "lazo"

        if index_file is not None:
            self.load_index(index_file)

        else:
            self.host = host
            self.port = port
            self.partition_size = partition_size

        self.lazo_client = lazo_index_service.LazoIndexClient(
            host=self.host, port=self.port
        )

        if data_dir is not None:
            data_dir = Path(data_dir)
            self.add_tables_from_path(data_dir)
            self.data_dir = data_dir

    def _index_single_table(self, df: pl.DataFrame, tab_name: str):
        jd_logger.debug("STARTING: Tab %s" % tab_name)
        for col in df.select(cs.string()).columns:
            partitions = self._partition_list_for_indexing(df[col].unique().to_list())
            for partition in partitions:
                try:
                    (
                        n_permutations,
                        hash_values,
                        cardinality,
                    ) = self.lazo_client.index_data(partition, tab_name, col)
                except LazoError as e:
                    print(e)
                    print(tab_name, col)
                    jd_logger.error("FAILURE: tab %s col %s " % (tab_name, col))
                    sh_logger.error("FAILURE: tab %s col %s " % (tab_name, col))
                    continue
        jd_logger.debug("SUCCESS: Tab %s " % tab_name)

    def _partition_list_for_indexing(self, value_list: list):
        size_of_list = sum([getsizeof(val) for val in value_list]) + getsizeof(
            value_list
        )

        # Taking smaller partitions to try to avoid OOM error
        n_partitions = size_of_list // LAZO_MESSAGE_SIZE_LIMIT + 4
        partitions = [
            list(a)
            for a in np.array_split(np.array(value_list, dtype=str), n_partitions)
        ]

        return partitions

    def add_tables_from_dict(self, df_dict: dict):
        """Add all tables in the provided dictionary to the index.

        The dict should have format {tab_name : df}.

        Args:
            df_dict (dict): Dictionary containing all the tables.
        """
        for tab_name, df in df_dict.items():
            self._index_single_table(df, tab_name)

    def add_tables_from_path(self, data_path):
        """Add tables to the index reading metadata from the given `data_path`.
        `data_path` should contain json files that include the metadata of the file.

        Args:
            data_path (Path): Path to the directory to scan for metadata.
        """
        if not data_path.exists():
            raise IOError("Invalid data directory")

        total_files = sum(1 for f in data_path.glob("*.json"))

        if total_files == 0:
            raise RuntimeError(f"No json files found in {data_path}.")

        for path in tqdm(
            data_path.glob("*.json"),
            total=total_files,
            leave=False,
            desc="Adding tables to index",
        ):
            mdata_dict = json.load(open(path, "r"))
            ds_hash = mdata_dict["hash"]
            df = pl.read_parquet(mdata_dict["full_path"])
            self._index_single_table(df, ds_hash)

    def add_single_table(self, df: pl.DataFrame, tab_name: str):
        """Add a single table to the index.

        Args:
            df (pl.DataFrame): Table to index.
            tab_name (str): Table name.
        """
        print(tab_name)
        self._index_single_table(df, tab_name)

    def query_index(self, query):
        """Query the index with the given `query`.
        Note that if the size of the query is > `partition_size`, it will be
        clamped to `partition_size` after removing duplicates.

        Args:
            query (Iterable): Query values.

        Returns:
            _type_: Results of the query, which include the table name, column name and similarity score.
        """
        # Note that I am clamping the query to the first `partition_size` elements
        query_part = self._partition_list_for_indexing(query)[0]
        query_results = self.lazo_client.query_data(query_part)

        return query_results

    def clean_index(self, cleanup_dict: dict):
        for tab_name, column_list in cleanup_dict.items():
            self.lazo_client.remove_sketches(tab_name, column_list)

    def load_index(self, index_file):
        if Path(index_file).exists():
            with open(index_file, "rb") as fp:
                params = load(fp)
                self.host = params["host"]
                self.port = params["port"]
                self.partition_size = params["partition_size"]
        else:
            raise FileNotFoundError(f"File {index_file} not found.")

    def save_index(self, output_path):
        out_dict = {
            "index_name": self.index_name,
            "host": self.host,
            "port": self.port,
            "partition_size": self.partition_size,
        }
        with open(output_path, "wb") as fp:
            dump(out_dict, fp)


class CountVectorizerIndex:
    def __init__(
        self,
        data_lake_path=None,
        base_table_path=None,
        query_column=None,
        binary=False,
        file_path=None,
        n_jobs=1,
    ) -> None:
        if file_path is not None:
            if Path(file_path).exists():
                with open(file_path, "wb") as fp:
                    mdata = pickle.load(fp)
                    self.data_lake_path = mdata["base_path"]
                    self.base_table = mdata["base_table"]
                    self.query_column = mdata["query_column"]
                    self.binary = mdata["binary"]
                    self.keys = mdata["keys"]
                    self.n_jobs = 0
                self.count_vectorizer = None
                self.count_matrix = self.load_count_matrix(mdata["path_count_matrix"])
            else:
                raise FileNotFoundError
        else:
            if any([base_table is None, query_column is None, data_lake_path is None]):
                raise ValueError
            if not Path(data_lake_path).exists() or not (
                Path(base_table_path).exists()
            ):
                raise FileNotFoundError

            self.data_lake_path = data_lake_path
            self.base_table_path = base_table_path
            self.base_table = pl.read_parquet(base_table_path)
            self.query_column = query_column
            self.binary = binary
            self.count_vectorizer = CountVectorizer(
                token_pattern=r"(?u)(<[\S]+>)[ ]{4}", binary=binary
            )
            self.n_jobs = n_jobs
            self.count_matrix, self.keys = self.build_count_matrix()

    def _prepare_single_table(self, table_path):
        table = pl.read_parquet(table_path)
        table_name = table_path.stem
        res = []
        sep = " " * 4
        for col in table.select(cs.string()).columns:
            values = sep.join([_ for _ in table[col].to_list() if _ is not None])
            values += " " * 4
            key = f"{table_name}__{col}"
            res.append((key, values))
        return res

    def _get_values(self, list_values):
        s = "    ".join(list_values) + "    "
        return [s]

    def build_count_matrix(self):
        values = self._get_values(self.base_table[self.query_column].to_list())
        self.count_vectorizer.fit(values)

        partial_result = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._prepare_single_table)(pth)
            for pth in self.data_lake_path.glob("*.parquet")
        )
        result = [
            col_tup for table_result in partial_result for col_tup in table_result
        ]
        keys = np.array([_[0] for _ in result])
        return (
            self.count_vectorizer.transform([col_tup[1] for col_tup in result]),
            keys,
        )

    def load_count_matrix(self, path_matrix):
        if Path(path_matrix).exists():
            mat = load_npz(path_matrix)
            return mat

        raise FileNotFoundError

    def save_count_matrix(self, path_matrix):
        save_npz(path_matrix, self.count_matrix)

    def query_index(self, query_column=None, top_k=200):
        # NOTE: `query_column` is ignored, but it is kept as argument for compatibility
        # with other code
        sum_res = np.array(self.count_matrix.sum(axis=1)).ravel()
        s_index = np.flip(sum_res.argsort())
        split_keys = [_.split("__") for _ in np.flip(self.keys[s_index])]
        voc_size = self.count_matrix.shape[1]
        # TODO: add more a more clever way of defining the ranking
        ranking = [
            a + [b / voc_size] for a, b in zip(split_keys, sum_res[s_index]) if b > 0
        ]
        if top_k == -1:
            return ranking
        else:
            return ranking[:top_k]

    def save_index(self, output_path):
        path_count_matrix = Path()
        dd = {
            "base_path": self.data_lake_path,
            "base_table_path": self.base_table_path,
            "query_column": self.query_column,
            "binary": self.binary,
            "keys": self.keys,
            "path_count_matrix": path_count_matrix,
        }
        with open(output_path, "wb") as fp:
            dump(dd, fp)

        self.save_count_matrix(path_count_matrix)
