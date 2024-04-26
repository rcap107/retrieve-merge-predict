import json
from pathlib import Path
from sys import getsizeof

import lazo_index_service
import numpy as np
import polars as pl
import polars.selectors as cs
from datasketch import LeanMinHash, MinHash, MinHashLSHEnsemble
from joblib import Parallel, delayed, dump, load
from lazo_index_service.errors import LazoError
from memory_profiler import profile
from scipy.sparse import csr_array, load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import murmurhash3_32
from tqdm import tqdm

LAZO_MESSAGE_SIZE_LIMIT = 4194304
MAX_SIZE = np.iinfo(np.uint32).max


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
        metadata_dir: str | Path = None,
        thresholds: int | list = 20,
        num_perm: int = 128,
        num_part: int = 32,
        oneshot: bool = True,
        index_file: str | Path = None,
        n_jobs=1,
        no_tag: bool = True,
    ) -> None:
        """
        If `index_file` is provided, the data structures required for the index are loaded from the given
        index file.

        If `oneshot` is set to True, the index will be initialized within this function.
        If `oneshot` is set to False, the index creation will not be wrapped up until the user manually
        invokes `create_ensembles`: this allows to update the indices with tables that were not added
        while scanning `metadata_dir`.

        Args:
            metadata_dir (str, optional): Path to the dir that contains the metadata of the target tables.
            thresholds (int | list, optional): Threshold or list of thresholds to be used by the ensemble. Defaults to 20.
            num_perm (int, optional): Number of hash permutations. Defaults to 128.
            num_part (int, optional): Number of partitions. Defaults to 32.
            oneshot (bool, optional): If False, index will have to be finalized by the user. Defaults to True.
            index_file (str, optional): Path to a file containing a pre-computed index.
        """
        self.index_name = "minhash"

        self.hash_index = []
        self.num_perm = num_perm
        self.num_part = num_part
        self.no_tag = no_tag
        if isinstance(thresholds, list):
            self.thresholds = sorted(thresholds)
        else:
            self.thresholds = [thresholds]
        self.single_threshold = len(self.thresholds) == 1
        self.initialized = False
        self.ensembles = {}
        self.n_jobs = n_jobs

        if index_file is not None:
            self.load_index(index_file)
            self.initialized = True

        elif metadata_dir is not None:
            self.metadata_dir = Path(metadata_dir)
            if not self.metadata_dir.exists():
                raise IOError("Invalid data directory")

            self.add_tables_from_path(self.metadata_dir)

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
            print("Creating ensembles.")
            for t in tqdm(self.thresholds):
                ens = MinHashLSHEnsemble(
                    threshold=t / 100, num_perm=self.num_perm, num_part=self.num_part
                )
                ens.index(self.hash_index)
                self.ensembles[t] = ens
            print("Ensembles creation complete. ")
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

    def query_index(self, query, threshold=None, to_dataframe=False, top_k=200):
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
                if any(th not in self.ensembles for th in threshold):
                    raise ValueError("Invalid thresholds in the provided list.")
                for th in sorted(threshold):
                    ens = self.ensembles[th]
                    res = list(ens.query(m_query, len(query)))
                    query_dict.update(self.prepare_result(res, threshold))
            else:
                for th, ens in self.ensembles.items():
                    res = list(ens.query(m_query, len(query)))
                    query_dict.update(self.prepare_result(res, th))

            if to_dataframe:
                query_results = pl.from_records(
                    query_dict, schema=["hash", "column", "threshold"]
                )
            else:
                query_results = []
                for k, v in query_dict.items():
                    query_results.append((k[0], k[1], v))
            if top_k == -1:
                return query_results
            return query_results[:top_k]
        else:
            raise RuntimeError("Ensembles are not initialized.")

    def save_ensembles(self, output_path):
        """Save the ensembles to file in the given `output_path`.

        Args:
            output_path (str): Output path.
        """
        with open(output_path, "wb") as fp:
            dump(self.ensembles, fp)

    def save_index(self, output_dir: str | Path):
        """Persist the index on disk in the given `output_dir`.

        Args:
            output_dir (str | Path): Path where the index will be saved.

        Returns:
            Path: Destination path of the saved index.
        """
        if self.no_tag:
            index_name = "minhash_index"
        else:
            if self.single_threshold:
                index_name = f"minhash_index_{self.thresholds[0]}"
            else:
                index_name = (
                    f"minhash_index_{'_'.join([str(_) for _ in self.thresholds])}"
                )
            self.index_name = index_name

        out_dict = {
            "index_name": self.index_name,
            "hash_index": self.hash_index,
            "num_perm": self.num_perm,
            "num_part": self.num_part,
            "thresholds": self.thresholds,
            "ensembles": self.ensembles,
        }

        index_path = Path(
            output_dir,
            index_name + ".pickle",
        )
        with open(
            index_path,
            "wb",
        ) as fp:
            dump(out_dict, fp)
        return index_path

    def load_index(
        self, index_file: str | Path | None = None, index_dict: dict | None = None
    ):
        if index_file is not None:
            if Path(index_file).exists():
                with open(index_file, "rb") as fp:
                    index_dict = load(fp)
                    self.hash_index = index_dict["hash_index"]
                    self.num_perm = index_dict["num_perm"]
                    self.num_part = index_dict["num_part"]
                    self.thresholds = index_dict["thresholds"]
                    self.ensembles = index_dict["ensembles"]
                    self.index_name = index_dict["index_name"]
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
                    continue

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

    def save_index(self, output_dir):
        out_dict = {
            "index_name": self.index_name,
            "host": self.host,
            "port": self.port,
            "partition_size": self.partition_size,
        }
        with open(Path(output_dir, "lazo_index.pickle"), "wb") as fp:
            dump(out_dict, fp)


class CountVectorizerIndex:
    def __init__(
        self,
        metadata_dir=None,
        base_table_path=None,
        query_column=None,
        binary=False,
        file_path=None,
        n_jobs=1,
    ) -> None:
        self.index_name = "count_vectorizer"
        self.sep = "|" * 4
        if file_path is not None:
            if Path(file_path).exists():
                with open(file_path, "rb") as fp:
                    mdata = load(fp)
                    self.data_dir = mdata["data_dir"]
                    self.base_table_path = mdata["base_table_path"]
                    self.query_column = mdata["query_column"]
                    self.binary = mdata["binary"]
                    self.keys = mdata["keys"]
                    self.n_jobs = 0
                    self.count_matrix = mdata["count_matrix"]
                self.count_vectorizer = None
            else:
                raise FileNotFoundError
        else:
            if any(
                [base_table_path is None, query_column is None, metadata_dir is None]
            ):
                raise ValueError
            if not Path(metadata_dir).exists() or not Path(base_table_path).exists():
                raise FileNotFoundError

            self.data_dir = Path(metadata_dir)
            self.base_table_path = Path(base_table_path)
            self.base_table = pl.read_parquet(base_table_path)
            self.query_column = query_column
            if query_column not in self.base_table.columns:
                raise pl.ColumnNotFoundError
            self.binary = binary
            self.count_vectorizer = CountVectorizer(
                token_pattern=r"(?u)([^|]*)[|]{4}", binary=binary
            )
            self.n_jobs = n_jobs
            self.count_matrix, self.keys = self._build_count_matrix()

    def _prepare_single_table(self, path):
        with open(path, "r") as fp:
            mdata_dict = json.load(fp)
        table_path = mdata_dict["full_path"]
        # Selecting only string columns
        table = pl.read_parquet(table_path).select(cs.string())
        ds_hash = mdata_dict["hash"]
        res = []
        for col in table.select(cs.string()).columns:
            values = self.sep.join([_ for _ in table[col].to_list() if _ is not None])
            values += self.sep
            key = f"{ds_hash}__{col}"
            res.append((key, values))
        return res

    def _get_values(self, list_values):
        s = self.sep.join(list_values) + self.sep
        return [s]

    def _build_count_matrix(self):
        values = self._get_values(self.base_table[self.query_column].to_list())
        self.count_vectorizer.fit(values)
        total = sum([1 for _ in self.data_dir.glob("*.json")])
        partial_result = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._prepare_single_table)(pth)
            for pth in tqdm(self.data_dir.glob("*.json"), total=total)
        )
        result = [
            col_tup for table_result in partial_result for col_tup in table_result
        ]
        keys = np.array([_[0] for _ in result])
        count_matrix = self.count_vectorizer.transform(
            [col_tup[1] for col_tup in result]
        )
        return (
            count_matrix,
            keys,
        )

    def _load_count_matrix(self, path_matrix):
        if Path(path_matrix).exists():
            mat = load_npz(path_matrix)
            return mat

        raise FileNotFoundError

    def _save_count_matrix(self, path_matrix):
        save_npz(path_matrix, self.count_matrix)

    def query_index(self, query_column=None, top_k=200):
        # NOTE: `query_column` is ignored, but it is kept as argument for compatibility
        # with other code
        non_zero_count = self.count_matrix.getnnz(axis=1)
        s_index = np.flip(non_zero_count.argsort())
        split_keys = [_.split("__") for _ in self.keys[s_index]]
        voc_size = self.count_matrix.shape[1]

        # This returns the contain
        ranking = [
            a + [b / voc_size]
            for a, b in zip(split_keys, non_zero_count[s_index])
            if b > 0
        ]
        if top_k == -1:
            return ranking
        return ranking[:top_k]

    def save_index(self, output_dir):
        path_mdata = Path(
            output_dir,
            f"cv_index_{self.base_table_path.stem}_{self.query_column}.pickle",
        )
        dd = {
            "index_name": self.index_name,
            "data_dir": self.data_dir,
            "base_table_path": self.base_table_path,
            "query_column": self.query_column,
            "binary": self.binary,
            "keys": self.keys,
            "count_matrix": self.count_matrix,
        }
        with open(path_mdata, "wb") as fp:
            dump(dd, fp, compress=True)


class ExactMatchingIndex:
    def __init__(
        self,
        metadata_dir: str | Path = None,
        base_table_path: str | Path = None,
        query_column: str = None,
        file_path: str | Path = None,
        n_jobs: int = 1,
    ) -> None:
        self.index_name = "exact_matching"
        if file_path is not None:
            if Path(file_path).exists():
                with open(file_path, "rb") as fp:
                    mdata = load(fp)
                    self.metadata_dir = mdata["metadata_dir"]
                    self.base_table_path = mdata["base_table_path"]
                    self.query_column = mdata["query_column"]
                    self.counts = mdata["counts"]
            else:
                raise FileNotFoundError
        else:
            if any(
                [base_table_path is None, query_column is None, metadata_dir is None]
            ):
                raise ValueError
            if not Path(metadata_dir).exists():
                raise FileNotFoundError
            if not Path(base_table_path).exists():
                raise FileNotFoundError

            self.metadata_dir = Path(metadata_dir)
            self.base_table_path = Path(base_table_path)
            self.base_table = pl.read_parquet(base_table_path)
            self.query_column = query_column
            if query_column not in self.base_table.columns:
                raise pl.ColumnNotFoundError
            self.n_jobs = n_jobs
            self.unique_base_table = set(
                self.base_table[query_column].unique().to_list()
            )

            self.unique_base_table = set(self.base_table[query_column].unique())
            self.counts = self._build_count_matrix(self.metadata_dir)

    @staticmethod
    def find_unique_keys(df, key_cols):
        """Find the set of unique keys given a combination of columns.

        This function is used to find what is the potential cardinality of a join key.

        Args:
            df (Union[pd.DataFrame, pl.DataFrame]): Dataframe to estimate key cardinality on.
            key_cols (list): List of key columns.

        Returns:
            _type_: List of unique keys.
        """
        try:
            unique_keys = df[key_cols].unique()
        except pl.DuplicateError:
            # Raise exception if a column name is duplicated
            unique_keys = None

        return unique_keys

    def _measure_containment(self, candidate_table: pl.DataFrame, right_on):
        unique_cand = self.find_unique_keys(candidate_table, right_on)

        s1 = self.unique_base_table
        s2 = set(unique_cand[right_on].to_series())
        return len(s1.intersection(s2)) / len(s1)

    # def _measure_containment(self, candidate_table: pl.DataFrame, right_on):
    #     cloned = self.unique_base_table.clone()
    #     return len(
    #         cloned.list.set_intersection(
    #             candidate_table.select(pl.col(right_on).unique()).to_series().implode()
    #         ).explode()
    #     ) / len(self.unique_base_table)

    def _prepare_single_table(self, fpath):
        overlap_dict = {}
        with open(fpath, "r") as fp:
            mdata = json.load(fp)
            cnd_path = mdata["full_path"]
            cnd_hash = mdata["hash"]

        df_cnd = pl.read_parquet(cnd_path)
        for col in df_cnd.select(cs.string()).columns:
            pair = (cnd_hash, col)
            cont = self._measure_containment(df_cnd, right_on=[col])
            overlap_dict[pair] = cont
        return overlap_dict

    def _build_count_matrix(self, mdata_path):
        # Building the pairwise distance with joblib
        r = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._prepare_single_table)(
                fpath,
            )
            for fpath in tqdm(
                mdata_path.glob("*.json"),
                position=0,
                leave=False,
                total=sum(1 for _ in mdata_path.glob("*.json")),
            )
        )

        overlap_dict = {key: val for result in r for key, val in result.items()}
        df_overlap = pl.from_dict(
            {
                "key": overlap_dict.keys(),
                "containment": overlap_dict.values(),
            }
        )
        df_overlap = (
            df_overlap.with_columns(
                pl.col("key").list.to_struct().struct.rename_fields(["hash", "col"])
            )
            .unnest("key")
            .sort("containment", descending=True)
        )
        return df_overlap

    def query_index(
        self,
        query_column=None,
        top_k=200,
    ):
        query_results = self.counts.filter(pl.col("containment") > 0).sort(
            "containment", descending=True
        )
        if top_k > 0:
            return query_results.top_k(top_k, by="containment").rows()
        else:
            return query_results.rows()

    def save_index(self, output_dir):
        path_mdata = Path(
            output_dir,
            f"em_index_{self.base_table_path.stem}_{self.query_column}.pickle",
        )
        dd = {
            "index_name": self.index_name,
            "metadata_dir": self.metadata_dir,
            "base_table_path": self.base_table_path,
            "query_column": self.query_column,
            "counts": self.counts,
        }
        with open(path_mdata, "wb") as fp:
            dump(dd, fp, compress=True)


class InvertedIndex:
    def __init__(
        self,
        metadata_dir: str | Path = None,
        file_path: str | Path = None,
        binary: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.index_name = "inverted_index"
        if file_path is not None:
            if Path(file_path).exists():
                with open(file_path, "rb") as fp:
                    # TODO: actually implement this method
                    raise NotImplementedError
            else:
                raise FileNotFoundError
        else:
            if metadata_dir is None:
                raise ValueError
            if not Path(metadata_dir).exists():
                raise FileNotFoundError

            self.data_dir = Path(metadata_dir)
            self.n_jobs = n_jobs
            self.binary = binary

            self.schema_mapping = self._prepare_schema_mapping()
            self.key_mapping = None

            self.keys, self.mat = self._prepare_data_structures()

    def _prepare_matrix(self, path: Path, col_keys: dict):
        with open(path, "r") as fp:
            mdata_dict = json.load(fp)
        table_path = mdata_dict["full_path"]
        # Selecting only string columns
        table = pl.read_parquet(table_path)
        if len(table) == 0:
            return None
        ds_hash = mdata_dict["hash"]
        table.columns = [f"{ds_hash}__{k}" for k in table.columns]

        data = []
        i_arr = []
        j_arr = []

        for col, key in col_keys.items():
            if not self.binary:
                raise NotImplementedError
                values, counts = table[col].drop_nulls().value_counts()
            else:
                values = table[col].drop_nulls().unique()
                counts = np.ones_like(values, dtype=np.uint8)

            # TODO: maybe I can keep this as a list and concatenate to lists
            i = np.array(
                [murmurhash3_32(val, positive=True) for val in values], dtype=np.uint32
            )

            j = key * np.ones_like(i, dtype=np.uint32)

            data.append(counts)
            i_arr.append(i)
            j_arr.append(j)

        # TODO: maybe this can be optimized?
        return (
            col_keys,
            (np.concatenate(data), (np.concatenate(i_arr), np.concatenate(j_arr))),
        )

    def _prepare_schema_mapping(self):
        schema_mapping = {}
        tot_col = 0
        print("Preparing schema mapping")
        for idx, p in enumerate(self.data_dir.glob("**/*.json")):
            with open(p, "r") as fp:
                mdata_dict = json.load(fp)
            table_path = mdata_dict["full_path"]
            ds_hash = mdata_dict["hash"]
            schema = pl.read_parquet_schema(table_path)
            schema = {f"{ds_hash}__{k}": v for k, v in schema.items() if v == pl.String}
            schema_mapping[p] = dict(
                zip(schema.keys(), range(tot_col, tot_col + len(schema.keys())))
            )
            tot_col += len(schema)

        return schema_mapping

    def _prepare_data_structures(self):
        r = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._prepare_matrix)(fpath, self.schema_mapping[fpath])
            for fpath in tqdm(
                self.data_dir.glob("**/*.json"),
                position=0,
                leave=False,
                total=sum(1 for _ in self.data_dir.glob("**/*.json")),
            )
        )
        result = list(filter(lambda x: x is not None, r))

        result = list(zip(*result))
        keys = {v: k for d in self.schema_mapping.values() for k, v in d.items()}

        mats = result[1]
        print("concatenating data")
        data = np.concatenate([m[0] for m in mats])
        print("concatenating coordinates")
        coordinates = np.concatenate([m[1] for m in mats], axis=1)
        i = coordinates[0, :]
        j = coordinates[1, :]
        # Creating a mapping between the hash and the index to reduce the memory footprint
        self.key_mapping = {k: idx for idx, k in enumerate(i)}
        i = [self.key_mapping[_] for _ in i]

        # Creating a csc matrix by concatenating the data entries
        print("Building csr array")
        mat = csr_array((data, (i, j)), dtype=np.uint32)

        return keys, mat

    def query_index(self, query: list, top_k: int = 200):
        dedup_query = set(query)
        q_in = [
            self.key_mapping.get(murmurhash3_32(_, positive=True), None)
            for _ in dedup_query
        ]
        q_in = set(filter(lambda x: x is not None, q_in))
        r = np.array(self.mat[list(q_in)].sum(axis=0)).squeeze()
        r = r / len(dedup_query)
        res = [(self.keys[_], r[_]) for _ in np.flatnonzero(r)]
        df_r = pl.from_records(res, schema=["candidate", "similarity_score"])
        if top_k == -1:
            return df_r.sort("similarity_score", descending=True)
        else:
            return df_r.top_k(by="similarity_score", k=top_k)


class StarmieWrapper:
    def __init__(
        self,
        import_path: str | Path = None,
        base_table_path: str | Path = None,
        file_path: str | Path = None,
    ) -> None:
        self.index_name = "starmie"
        if file_path is not None:
            if Path(file_path).exists():
                with open(file_path, "rb") as fp:
                    mdata = load(fp)
                    self.base_table_path = Path(mdata["base_table_path"])
                    self.ranking = mdata["ranking"]
            else:
                raise FileNotFoundError(f"STARMIE index file {file_path} not found.")
        elif import_path is not None:
            if Path(import_path).exists():
                import_df = pl.read_parquet(import_path)
                import_df = import_df.with_columns(
                    pl.col("join_columns")
                    .list.to_struct()
                    .struct.rename_fields(["left_on", "right_on"])
                ).unnest("join_columns")
                self.ranking = import_df.clone()
                self.base_table_path = Path(base_table_path)
            else:
                raise FileNotFoundError(f"Import file {import_path} not found.")
        else:
            raise ValueError("Either import_path or file_path must be provided.")

    def query_index(
        self,
        query_column: str = None,
        top_k: int = 200,
    ):
        if query_column is None:
            raise ValueError("Invalid value provided for query_column")
        query_results = (
            self.ranking.filter(
                (pl.col("left_on") == query_column) & (pl.col("similarity") > 0)
            )
            .sort("similarity", descending=True)
            .drop("left_on")
        )

        if top_k > 0:
            return query_results.top_k(top_k, by="similarity").rows()
        return query_results.rows()

    def save_index(self, output_dir: str | Path):
        path_mdata = Path(
            output_dir,
            f"starmie_index-{self.base_table_path.stem}.pickle",
        )
        print(f"{path_mdata}")
        dd = {
            "index_name": self.index_name,
            "base_table_path": self.base_table_path,
            "ranking": self.ranking,
        }
        with open(path_mdata, "wb") as fp:
            dump(dd, fp, compress=True)
