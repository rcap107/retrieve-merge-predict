#%%
from pathlib import Path

import polars as pl

from src.data_structures.retrieval_methods import InverseIndex, InverseIndex2

#%%
pth = Path("data/metadata/binary_update")
table = pl.read_parquet(
    Path("data/source_tables/yadl/us_elections-yadl-depleted.parquet")
)
query = table["col_to_embed"].unique().to_numpy()
# %%
# ii = InverseIndex2(pth)
ii = InverseIndex(pth)
# %%

# %%
ii.query_index(query)
