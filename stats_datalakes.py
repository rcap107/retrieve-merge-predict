# %%
from pathlib import Path

import polars as pl
import polars.selectors as cs
from joblib import Parallel, delayed
from tqdm import tqdm

# %%
path_wordnet = Path("data/yadl/wordnet_full/")
path_wordnet_vldb = Path("data/yadl/wordnet_vldb/")
path_wordnet_vldb_wide = Path("data/yadl/wordnet_vldb_wide/")
path_binary = Path("data/yadl/binary_update/")
path_open_data = Path("data/open_data_us/")


# %%
def table_profile(table_path, data_lake):
    df = pl.read_parquet(table_path)
    n_num = df.select(cs.numeric()).shape[1]
    c_num = df.select(~cs.numeric()).shape[1]
    if len(df) > 0:
        avg_null = df.null_count().mean_horizontal().item() / len(df)
    else:
        avg_null = 0
    d = {
        "data_lake": data_lake,
        "table": table_path.stem,
        "num_attr": n_num,
        "cat_attr": c_num,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "avg_null": avg_null,
    }

    return d


# %%
def get_stats(df: pl.DataFrame):
    return df.select(
        pl.col("data_lake").first(),
        pl.col("n_tables").first(),
        pl.col("num_attr").mean().alias("mean_num_attr"),
        pl.col("num_attr").median().alias("median_num_attr"),
        pl.col("cat_attr").mean().alias("mean_cat_attr"),
        pl.col("cat_attr").median().alias("median_cat_attr"),
        pl.col("n_rows").mean().alias("mean_n_rows"),
        pl.col("n_rows").median().alias("median_n_rows"),
        pl.col("n_cols").mean().alias("mean_n_cols"),
        pl.col("n_cols").median().alias("median_n_cols"),
        pl.col("avg_null").mean().alias("mean_avg_null"),
        pl.col("avg_null").median().alias("median_avg_null"),
    )


def run_profile(dl_path, dl_name, n_jobs):
    total_ = sum(1 for _ in dl_path.glob("**/*.parquet"))
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    output = parallel(
        delayed(table_profile)(tab, dl_name)
        for tab in tqdm(
            dl_path.glob("**/*.parquet"),
            desc=dl_name,
            total=total_,
        )
    )
    profiles = list(output)
    return profiles


# %%
cases = [
    (path_wordnet_vldb, "vldb_flat"),
    (path_wordnet_vldb_wide, "vldb_wide"),
    (path_binary, "binary"),
    (path_wordnet, "wordnet"),
    (path_open_data, "open_data"),
]

# %%
stats = []
# %%
for case in cases:
    p = run_profile(*case, n_jobs=16)
    df = pl.from_dicts(p).with_columns(pl.lit(len(p)).alias("n_tables"))
    stats.append(df)
list_stats = [get_stats(d) for d in stats]
df_stats = pl.concat(list_stats)
df_stats.transpose(column_names="data_lake", include_header=True).write_csv(
    "stats_datalakes.csv"
)


#%%
stats_ = []
for case in cases[:1]:
    p = run_profile(*case, n_jobs=16)
    df = pl.from_dicts(p).with_columns(pl.lit(len(p)).alias("n_tables"))
    stats_.append(df)
list_stats_ = [get_stats(d) for d in stats_]
df_stats = pl.concat(list_stats)
df_stats.transpose(column_names="data_lake", include_header=True).write_csv(
    "stats_datalakes.csv"
)


#%%
s = [
    pl.from_dicts(pp).with_columns(pl.lit(len(profiles)).alias("n_tables"))
    for pp in stats
]

# %%
profiles = []
for tab in path_wordnet_vldb.glob("**/*.parquet"):
    d = table_profile(tab, "vldb")
    profiles.append(d)
df = pl.from_dicts(profiles).with_columns(pl.lit(len(profiles)).alias("n_tables"))
stats.append(get_stats(df))
list_stats = [get_stats(d) for d in stats]
df_stats = pl.concat(list_stats)
df_stats.transpose(column_names="data_lake", include_header=True).write_csv(
    "stats_datalakes.csv"
)
# %%


parallel = Parallel(n_jobs=2, return_as="generator")
output = parallel(
    delayed(table_profile)(tab, "vldb")
    for tab in tqdm(
        path_wordnet_vldb_wide.glob("**/*.parquet"),
        desc="vldb",
        total=sum(1 for _ in path_wordnet_vldb_wide.glob("**/*.parquet")),
    )
)
profiles = list(output)
