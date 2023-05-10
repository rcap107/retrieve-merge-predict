#%%
import polars as pl
import pickle
from src.table_integration.utils_joins import execute_join
from pathlib import Path

#%%
candidates = pickle.load(open("mh_result.pickle", "rb"))
for cand in candidates:
    print(cand)

# %%
data_path = Path("data/yago3-dl/wordnet")
tab_name = "yago_seltab_wordnet_movie"
tab_path = Path(data_path, f"{tab_name}.parquet")
df = pl.read_parquet(tab_path)

# %%
