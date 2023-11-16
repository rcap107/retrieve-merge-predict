# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


# %%
def convert_df(df_path):
    df = pl.read_parquet(df_path)
    threshold = df.select(pl.col("target").median()).item()

    df = df.with_columns(
        pl.when(pl.col("target") > threshold).then(1).otherwise(0).alias("target")
    )
    df.write_parquet(df_path.with_stem(df_path.stem + "-clf"))


# %%
tab_names = [
    "company-employees-yadl",
    "housing-open_data",
    "movies-yadl",
    "us-accidents-yadl",
    "us-elections-dems",
    "us-presidential-results-yadl",
]
base_path = Path("data/source_tables/")

for tname in tab_names:
    df_path = Path(base_path, tname + ".parquet")
    convert_df(df_path)
