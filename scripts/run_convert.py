from pathlib import Path
from types import SimpleNamespace

from batch_convert_csv_parquet import convert

d = {
    "input_folder": "/home/soda/rcappuzz/store3/metam/open_data_large/opendata_cleaned",
    "output_folder": "/home/soda/rcappuzz/store3/metam/open_data_large_pq",
    "n_jobs": -1,
    "wipe_old": True,
}

ns = SimpleNamespace(**d)

convert(ns)
