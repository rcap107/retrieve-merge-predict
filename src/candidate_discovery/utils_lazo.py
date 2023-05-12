import lazo_index_service
import polars as pl
import pandas as pd

from pathlib import Path
import numpy as np
import json


class LazoIndex:
    """This class implements a wrapper around the lazo index client. The lazo
    index server must be already running on the machine.
    """

    def __init__(self, data_dir, partition_size=50_000, host="localhost", port=15449):
        """Initialize the LazoIndex class.

        Args:
            df_dict (dict, optional): Dictionary containing all the tables to be indexed.
            If the dictionary is too big to fit in memory, tables can be added to the index
            one at a time. Defaults to None.
            partition_size (int, optional): Due to how protobuf works, column domains will be
            partitioned in lists of size `partition_size`. Defaults to 50_000.
            host (str, optional): Lazo server host address. Defaults to "localhost".
            port (int, optional): Lazo server port. Defaults to 15449.
        """
        self.index_name = "lazo"
        self.lazo_client = lazo_index_service.LazoIndexClient(host=host, port=port)
        self.partition_size = partition_size

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise IOError("Invalid data directory")

        self.add_tables_from_path(self.data_dir)

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
        total_files = sum(1 for f in data_path.glob("*.json"))

        for path in data_path.glob("*.json"):
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
