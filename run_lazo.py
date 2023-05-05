#%%
import lazo_index_service
import polars as pl
import pandas as pd

from pathlib import Path
import numpy as np

#%%
# Connecting to the Lazo server
SERVER_HOST="localhost"
SERVER_PORT=15449

lazo_client = lazo_index_service.LazoIndexClient(host=SERVER_HOST, port=SERVER_PORT)
# %%
def read_dataset(tgt_dataset: Path):
    if tgt_dataset.suffix == ".csv":
        df = pl.read_csv(tgt_dataset)
    else:
        df = pl.read_parquet(tgt_dataset)
    
    df = df.select(
        [
            pl.all().str.lstrip("<").str.rstrip(">"),
            
        ]
    )
    return df

# %%
data_path = Path("data/yago3-dl/wordnet")
# %%
tab_name = "yago_seltab_wordnet_movie"
tab_path = Path(data_path, f"{tab_name}.parquet")
df = read_dataset(tab_path)

#%%
df_dict = { tab_name: df}
# %%
all_movie_tables = data_path.glob("**/subtabs/yago_seltab_wordnet_movie/*.parquet")
for tab_path in all_movie_tables:
    tab_name = tab_path.stem
    print(tab_name)
    df_dict.update({tab_name: read_dataset(tab_path)})

#%%
def partition_list(value_list, partition_size=50000):
    n_partitions = len(value_list) // partition_size + 1
    partitions = [list(a) for a in np.array_split(np.array(value_list), n_partitions)]
    
    return partitions
#%%
for tab_name, tab in df_dict.items():
    print(tab_name)
    for col in tab.columns:
        partitions = partition_list(df[col].unique().to_list(), partition_size=50_000)
        for partition in partitions:
            (n_permutations, hash_values, cardinality) = lazo_client.index_data(
                partition, tab_name, col
            )

# %%
# Querying the index
for col in df.columns:
    query = df[col].sample(10000).to_list()
    print(col, lazo_client.query_data(query))
