#%%
from pathlib import Path

import polars as pl

# %%
for pth in Path("data/source_tables/batch/").glob("*.parquet"):
    print(pth)
    df = pl.read_parquet(pth)
    new_pth = pth.with_name(pth.stem + "-depleted" + pth.suffix)
    new_df = df.select("col_to_embed", "target")
    new_df.write_parquet(new_pth)
# %%
