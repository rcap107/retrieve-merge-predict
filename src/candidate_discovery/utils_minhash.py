import pickle
from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd
import polars as pl
from datasketch import MinHash, MinHashLSHEnsemble
from operator import itemgetter 
import json

class MinHashIndex:
    def __init__(
        self, data_dir, thresholds=[20], num_perm=128, num_part=32, oneshot=True
    ) -> None:
        """Index class based on `MinHashLSHEnsemble`. By default, it scans for metadata files
        in the provided `data_dir` and adds all them to the index.

        Since by default the LSHEnsemble queries based on a single threshold defined
        at creation time, this index builds accepts a list of thresholds and creates
        an ensemble for each.

        Ensembles do not support online updates, so after loading all tables in the index it is necessary
        to invoke the function `create_ensembles`. Querying without this step will raise an exception.
        
        If `oneshot` is set to True, the index will be initialized within this function. 
        If `oneshot` is set to False, the index creation will not be wrapped up until the user manually
        invokes `create_ensembles`: this allows to update the indices with tables that were not added
        while scanning `data_dir`.

        Args:
            data_dir (str): Path to the dir that contains the metadata of the target tables.
            thresholds (list, optional): List of thresholds to be used by the ensemble. Defaults to [20].
            num_perm (int, optional): Number of hash permutations. Defaults to 128.
            num_part (int, optional): Number of partitions. Defaults to 32.
            oneshot (bool, optional): If False, index will have to be finalized by the user. Defaults to True.
        """
        self.index_name = "minhash"
        self.hash_index = []
        self.num_perm = num_perm
        self.num_part = num_part
        self.thresholds = thresholds
        self.initialized = False
        self.ensembles = {}
        self.minhashes = {}

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise IOError("Invalid data directory")
        
        self.add_tables_from_path(self.data_dir)
        
        if oneshot:
            # If oneshot, wrap up the generation of the index here. If not, create_ensemble will have to be called later
            self.create_ensembles()

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

        for path in data_path.glob("*.json"):
            mdata_dict = json.load(open(path, "r"))
            ds_hash = mdata_dict["hash"]
            df = pl.read_parquet(mdata_dict["full_path"])
            self.add_single_table(df, ds_hash)
        

    def add_tables_from_dict(self, df_dict):
        """Given a dictionary of pl.DataFrames, generate minhashes for each dataframe.

        Args:
            df_dict (dict[str: pl.DataFrame]): Dictionary of dataframes.
        """
        t_dict = {}
        for tab_name, df in df_dict.items():
            # print(tab_name)
            t_dict.update(self._index_single_table(df, tab_name))

        self.minhashes.update(t_dict)
        self.hash_index += [(key, m, setlen) for key, (m, setlen) in t_dict.items()]

    def add_single_table(self, df: pl.DataFrame, tab_name):
        """Add a single table to the minhash dictionary.


        Args:
            df (pl.DataFrame): _description_
            tab_name (_type_): _description_
        """
        # print(tab_name)
        t_dict = self._index_single_table(df, tab_name)
        self.minhashes.update(t_dict)
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
        r = []
        for result in query_result:
            t, c = result.split("__")
            tup = (t, c, threshold)
            r.append(tup)
        return r

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
            query_results = []

            if threshold is not None:
                if any([th not in self.ensembles for th in threshold]):
                    raise ValueError(f"Invalid thresholds in the provided list.")
                else:
                    for th in threshold:                        
                        ens = self.ensembles[th]
                        res = list(ens.query(m_query, len(query)))
                        query_results += self.prepare_result(res, threshold)
            else:
                for threshold, ens in self.ensembles.items():
                    res = list(ens.query(m_query, len(query)))
                    query_results += self.prepare_result(res, threshold)

            if to_dataframe:
                query_results = pl.from_records(
                    query_results, schema=["hash", "column", "threshold"]
                )

            return query_results
        else:
            raise RuntimeError("Ensembles are not initialized.")

    def save_ensembles_to_pickle(self, output_path):
        """Save the ensembles to file in the given `output_path` as a pickle.

        Args:
            output_path (str): Output path.
        """
        pickle.dump(self.ensembles, open(output_path, "wb"))


# %%
