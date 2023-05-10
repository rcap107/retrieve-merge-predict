import polars as pl
import pandas as pd
import json
from pathlib import Path

def prepare_metadata_single_table(table_full_path):
    if self.path.suffix == ".csv":
        #TODO Add parameters for the `pl.read_csv` function
        return pl.read_csv(self.path)
    elif self.path.suffix == ".parquet":
        #TODO Add parameters for the `pl.read_parquet` function
        return pl.read_parquet(self.path)
    else:
        raise IOError(f"Extension {self.path.suffix} not supported.")
