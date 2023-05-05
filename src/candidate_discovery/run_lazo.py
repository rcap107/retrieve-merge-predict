#%%
import lazo_index_service
import polars as pl
import pandas as pd

from pathlib import Path
import random
import string

#%%
# Connecting to the Lazo server
SERVER_HOST="localhost"
SERVER_PORT=15449

lazo_client = lazo_index_service.LazoIndexClient(host=SERVER_HOST, port=SERVER_PORT)

#%%
# Used only for debugging
def generate_random_sequence(len_entry, len_sequence):
    return ["".join(random.choices(string.ascii_letters, k=len_entry)) for _ in range(len_sequence)]

# %%
pth = Path("data/yago3-dl/seltab/csv")
# %%
def read_dataset(tgt_dataset):
    df = pl.read_csv(tgt_dataset)
    df = df.select(
        [
            pl.all().str.lstrip("<").str.rstrip(">"),
            
        ]
    )

    return df
#%% 
for dataset_path in pth.iterdir():
    df = read_dataset(dataset_path)
    dsname = dataset_path.stem
    for col in df.columns:
        (n_permutations, hash_values, cardinality) = lazo_client.index_data(
            df[col].to_list(), dsname, col
        )

#%%    
dsname = "yago_seltab_wordnet_organization"
tgt_dataset = Path(pth, f"{dsname}.csv")
df = read_dataset(tgt_dataset)

for col in df.columns:
    (n_permutations, hash_values, cardinality) = lazo_client.index_data(
        df[col].to_list(), dsname, col
    )
#%%
dsname = "yago_seltab_wordnet_airport"
tgt_dataset = Path(pth, f"{dsname}.csv")
df = read_dataset(tgt_dataset)

for col in df.columns:
    (n_permutations, hash_values, cardinality) = lazo_client.index_data(
        df[col].to_list(), dsname, col
    )

# %%
# Querying the index
for col in df.columns:
    query = df[col].sample(10000).to_list()
    print(col, lazo_client.query_data(query))

#%%


# %%
(n_permutations, hash_values, cardinality) = lazo_client.index_data(
    df["subject"].to_list(), dsname, "subject"
)

# %%
# glob all parquet files in the path

list(Path("data/yago3-dl/seltab/subtabs/").glob("**/*.parquet"))