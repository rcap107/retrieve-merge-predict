# %%
import polars as pl
from pathlib import Path
from src.candidate_discovery.utils_minhash import MinHashIndex
from src.candidate_discovery.utils_lazo import LazoIndex
import pickle
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
mh_index.create_ensembles()

# %%
query = df["isLocatedIn"].sample(50000).drop_nulls()
mh_result = mh_index.query_index(query)

#%%
lazo_index = LazoIndex(df_dict)
# %%
lazo_result = lazo_index.query_index(query)
# %%
pickle.dump({k: v for k,v in enumerate(mh_result)}, open("mh_result.pickle", "wb"))
pickle.dump({k: v for k,v in enumerate(lazo_result)}, open("lazo_results.pickle", "wb"))

# %%
