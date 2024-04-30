#%%
# %cd ~/bench
# %%
from pathlib import Path

import polars as pl
import polars.selectors as cs

cfg = pl.Config()
cfg.set_fmt_str_lengths(150)

#%%
list_d = []
for pth in Path("data/source_tables/yadl").iterdir():
    df = pl.read_parquet(pth)
    n_num = df.select(cs.numeric()).shape[1]
    c_num = df.select(~cs.numeric()).shape[1]
    d = {"table": pth.stem, "num_attr": n_num, "cat_attr": c_num, "n_rows": len(df)}
    list_d.append(d)
# %%
df_stats = pl.from_dicts(list_d)
# %%
df_stats.write_csv("stats_tables_yadl.csv")
# %%
list_d = []
for pth in Path("data/source_tables/open_data_us").iterdir():
    if pth.suffix == ".parquet":
        df = pl.read_parquet(pth)
        n_num = df.select(cs.numeric()).shape[1]
        c_num = df.select(~cs.numeric()).shape[1]
        d = {"table": pth.stem, "num_attr": n_num, "cat_attr": c_num, "n_rows": len(df)}
        list_d.append(d)
# %%
df_stats = pl.from_dicts(list_d)
df_stats.write_csv(open("stats_tables_open_data.csv", "w"))

# %%
