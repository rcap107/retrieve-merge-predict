#%% 
# %load_ext autoreload
# %autoreload 2
from src.data_preparation.data_structures import RawDataset
# %%
import pandas as pd
import polars as pl
from pathlib import Path

# %%
data_folder = Path("data/yago3-dl/wordnet")
src_dataset_path = Path(data_folder, "")
dataset_path  = Path(src_dataset_path, "yago_seltab_wordnet_site.parquet")
ds = RawDataset(dataset_path, "yago3-dl", "data/metadata")
# %%
# ds.save_metadata_to_json()

# %%

ds.prepare_hash("write_json")

ds.prepare_hash("write_ipc")
 
ds.prepare_hash("zlib")

# %%

data_folder = Path("data/source_tables/ken_datasets")
src_dataset_path = Path(data_folder, "us-accidents")
dataset_path  = Path(src_dataset_path, "us-accidents.csv")
ds = RawDataset(dataset_path, "ken_datasets", "data/metadata")

# %%
%%timeit
ds.prepare_hash("write_json")

#%%
%%timeit
ds.prepare_hash("write_ipc")

#%%
%%timeit 
ds.prepare_hash("zlib")

# %%
