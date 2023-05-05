# %%
from datasketch import MinHashLSHEnsemble, MinHash
import polars as pl
from pathlib import Path
from src.candidate_discovery.minhash import MinHashIndex

# %%
data_path = Path("data/yago3-dl/wordnet")
# %%
tab_name = "yago_seltab_wordnet_movie"
tab_path = Path(data_path, f"{tab_name}.parquet")
df = pl.read_parquet(tab_path)

df_dict = { tab_name: df}
# %%
all_movie_tables = data_path.glob("**/subtabs/yago_seltab_wordnet_movie/*.parquet")
for tab_path in all_movie_tables:
    tab_name = tab_path.stem
    print(tab_name)
    df_dict.update({tab_name: pl.read_parquet(tab_path)})
#%%
mh_index = MinHashIndex(df_dict)

#%%
mh_index.create_ensembles()

# %%
query = df["isLocatedIn"].sample(50000).drop_nulls()
mh_index.query_ensembles(query)
# %%