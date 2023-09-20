import json
import logging
import pickle
from pathlib import Path

import lazo_index_service
import numpy as np
import polars as pl
from datasketch import MinHash, MinHashLSHEnsemble
from joblib import Parallel, delayed
from tqdm import tqdm

mh_logger = logging.getLogger("metadata_logger")


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


class ManualIndex(BaseIndex):
    """_summary_"""

    def __init__(
        self,
        index_file=None,
        match_list=None,
        df_base=None,
        tab_name=None,
        mdata_path=None,
        n_jobs=1,
    ):
        self.index_name = "manual"
        self.df = None
        self.df_base = df_base
        self.tab_name = tab_name

        if index_file is not None or match_list is not None:
            if index_file is not None:
                if Path(index_file).exists():
                    self.load_index(index_file)
                else:
                    raise FileNotFoundError(
                        f"Manual index file {index_file} not found."
                    )
            elif match_list is not None:
                self.df = match_list
            if not all((col in self.df.columns for col in ["hash", "col", "overlap"])):
                raise ValueError(
                    "Some required columns were not found in the provided file."
                )
        elif df_base is not None and mdata_path is not None:
            print("Building data structure.")
            if not Path(mdata_path).exists():
                raise FileNotFoundError(f"The provided mdata path is invalid.")
            self.df = self.measure_exact_overlap(df_base, mdata_path, n_jobs)
        else:
            # Do nothing
            pass

    def query_index(self, col_to_embed, top_k=10, threshold=0.1):
        result = (
            self.df.filter(pl.col("overlap") > threshold)
            .sort("overlap", descending=True)
            .limit(top_k)
            .to_numpy()
        )
        return [tuple(v) for v in result]

    @staticmethod
    def _evaluate_one_table(fpath, df_base):
        from src.methods.profiling import measure_containment

        overlap_dict = {}
        with open(fpath) as fp:
            mdata = json.load(fp)
            cnd_path = mdata["full_path"]
            cnd_hash = mdata["hash"]
            df_cnd = pl.read_parquet(cnd_path)
            for col in df_cnd.columns:
                pair = (cnd_hash, col)
                cont = measure_containment(
                    df_base, df_cnd, left_on=["col_to_embed"], right_on=[col]
                )
                overlap_dict[pair] = cont
        return overlap_dict

    def measure_exact_overlap(self, df_base, mdata_path, n_jobs):
        # Building the pairwise distance with joblib
        r = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self._evaluate_one_table)(fpath, df_base)
            for idx, fpath in tqdm(
                enumerate(mdata_path.glob("*.json")),
                position=0,
                leave=False,
                total=sum((1 for _ in mdata_path.glob("*.json"))),
            )
        )

        overlap_dict = {key: val for result in r for key, val in result.items()}
        df_overlap = pl.from_dict(
            {"key": list(overlap_dict.keys()), "overlap": list(overlap_dict.values())}
        )
        df_overlap = df_overlap.with_columns(
            pl.col("key").list.to_struct().struct.rename_fields(["hash", "col"])
        ).unnest("key")
        df_overlap = df_overlap.sort("overlap", descending=True)

        return df_overlap

    def save_index(self, output_path):
        out_dict = {
            "index_name": self.index_name,
            "df_overlap": self.df,
            "tab_name": self.tab_name,
        }
        output_path = Path(output_path).with_stem(f"{self.index_name}_{self.tab_name}")
        with open(output_path, "wb") as fp:
            pickle.dump(out_dict, fp)

    def load_index(self, index_path):
        if Path(index_path).exists():
            with open(index_path, "rb") as fp:
                in_dict = pickle.load(fp)
                self.df = in_dict["df_overlap"]
                self.tab_name = in_dict["tab_name"]
        else:
            raise FileNotFoundError(f"File {index_path} not found.")


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
            index_file (str, optional): Path to a pickle containing a pre-computed index.
        """
        self.index_name = "minhash"

        self.hash_index = []
        self.num_perm = num_perm
        self.num_part = num_part
        self.thresholds = sorted(thresholds)
        self.initialized = False
        self.ensembles = {}

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

    def _index_single_table(self, df: pl.DataFrame, tab_name) -> dict:
        """Generate the minhashes for a single dataframe.

        Args:
            df (pl.DataFrame): The input dataframe.
            tab_name (str): The name of the table, used for indexing.

        Returns:
            minhashes (dict): The minhashes generated for the given table.
        """
        minhashes = {}
        for col in df.columns:
            key = tab_name + "__" + col
            m = MinHash(num_perm=self.num_perm)
            uniques = df[col].drop_nulls().unique().cast(str)
            for u in uniques:
                m.update(u.encode("utf8"))
            minhashes[key] = (m, len(uniques))
        return minhashes

    def add_tables_from_path(self, data_path):
        """Add tables to the index reading metadata from the given `data_path`.
        `data_path` should contain json files that include the metadata of the file.

        Args:
            data_path (Path): Path to the directory to scan for metadata.
        """
        total_files = sum(1 for f in data_path.glob("*.json"))

        if total_files > 0:
            for path in tqdm(
                data_path.glob("*.json"),
                total=total_files,
                desc="Loading metadata in index",
            ):
                mdata_dict = json.load(open(path, "r"))
                ds_hash = mdata_dict["hash"]
                df = pl.read_parquet(mdata_dict["full_path"])
                try:
                    self.add_single_table(df, ds_hash)
                except AttributeError:
                    mh_logger.error("Error with file %s", str(mdata_dict["full_path"]))
        else:
            raise RuntimeError("No metadata files were found.")

    def add_tables_from_dict(self, df_dict):
        """Given a dictionary of pl.DataFrames, generate minhashes for each dataframe.

        Args:
            df_dict (dict[str: pl.DataFrame]): Dictionary of dataframes.
        """
        t_dict = {}
        for tab_name, df in df_dict.items():
            # print(tab_name)
            t_dict.update(self._index_single_table(df, tab_name))

        # self.minhashes.update(t_dict)
        self.hash_index += [(key, m, setlen) for key, (m, setlen) in t_dict.items()]

    def add_single_table(self, df: pl.DataFrame, tab_name):
        """Add a single table to the minhash dictionary.


        Args:
            df (pl.DataFrame): _description_
            tab_name (_type_): _description_
        """
        t_dict = self._index_single_table(df, tab_name)
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
            table, column = result.split("__")

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

    def save_ensembles_to_pickle(self, output_path):
        """Save the ensembles to file in the given `output_path` as a pickle.

        Args:
            output_path (str): Output path.
        """
        pickle.dump(self.ensembles, open(output_path, "wb"))

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
            pickle.dump(out_dict, fp)

    def load_index(self, index_file=None, index_dict=None):
        if index_file is not None:
            if Path(index_file).exists():
                with open(index_file, "rb") as fp:
                    index_dict = pickle.load(fp)
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
        for col in df.columns:
            partitions = self._partition_list_for_indexing(
                df[col].unique().to_list(), partition_size=self.partition_size
            )
            for partition in partitions:
                (
                    n_permutations,
                    hash_values,
                    cardinality,
                ) = self.lazo_client.index_data(partition, tab_name, col)

    def _partition_list_for_indexing(self, value_list: list, partition_size: int):
        n_partitions = len(value_list) // partition_size + 1
        partitions = [
            list(a) for a in np.array_split(np.array(value_list), n_partitions)
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
        query_part = self._partition_list_for_indexing(query, self.partition_size)[0]
        query_results = self.lazo_client.query_data(query_part)

        return query_results

    def clean_index(self, cleanup_dict: dict):
        for tab_name, column_list in cleanup_dict.items():
            self.lazo_client.remove_sketches(tab_name, column_list)

    def load_index(self, index_file):
        if Path(index_file).exists():
            with open(index_file, "rb") as fp:
                params = pickle.load(fp)
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
            pickle.dump(out_dict, fp)
