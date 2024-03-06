#%%
from pathlib import Path

import polars as pl

from src.data_structures.retrieval_methods import InvertedIndex

#%%
pth = Path("data/metadata/wordnet_full")
table = pl.read_parquet(
    Path("data/source_tables/yadl/us_elections-yadl-depleted.parquet")
)
query = table["col_to_embed"].unique().to_numpy()
# %%
ii = InvertedIndex(pth, n_jobs=16)
# %%

# %%
ii.query_index(query)
