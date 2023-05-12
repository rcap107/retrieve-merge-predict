# %%
import polars as pl
from pathlib import Path
from src.candidate_discovery.utils_minhash import MinHashIndex
from src.candidate_discovery.utils_lazo import LazoIndex
import pickle
import json
from tqdm import tqdm

# %%
data_path = Path("data/yago3-dl/wordnet")
metadata_path = Path("data/metadata")
df_dict = {}

# %%
mh_index = MinHashIndex()

# Count the number of total files in the glob
total_files = sum(1 for f in metadata_path.glob("*.json"))

for path in tqdm(metadata_path.glob("*.json"), total=total_files):
    mdata_dict = json.load(open(path, "r"))
    ds_hash = mdata_dict["hash"]
    df = pl.read_parquet(mdata_dict["full_path"])
    mh_index.add_single_table(df, ds_hash)


# %%
mh_index.create_ensembles()

# Saving the file
pickle.dump(obj=mh_index, file=open("mh_index.pickle", "wb"))

# %%
if False:
    df_dict = {tab_name: df}
    all_movie_tables = data_path.glob("**/subtabs/yago_seltab_wordnet_movie/*.parquet")
    for tab_path in all_movie_tables:
        tab_name = tab_path.stem
        print(tab_name)
        df_dict.update({tab_name: pl.read_parquet(tab_path)})
    mh_index = MinHashIndex(df_dict)
    mh_index.create_ensembles()
# %%
tab_name = "yago_seltab_wordnet_movie"
tab_path = Path(data_path, f"{tab_name}.parquet")
df = pl.read_parquet(tab_path)

# %%
query = df["isLocatedIn"].sample(50000).drop_nulls()
mh_result = mh_index.query_index(query)
pickle.dump({k: v for k, v in enumerate(mh_result)}, open("mh_result.pickle", "wb"))


############################ LAZO
# %%
lazo_index = LazoIndex(df_dict)
# %%
lazo_result = lazo_index.query_index(query)
# %%
pickle.dump(
    {k: v for k, v in enumerate(lazo_result)}, open("lazo_results.pickle", "wb")
)

# %%
# Load index from pickle
ensembles = pickle.load(open("mh_index.pickle", "rb"))

# %%
