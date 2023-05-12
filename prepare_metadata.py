#%% 
%load_ext autoreload
%autoreload 2
from src.data_preparation.data_structures import RawDataset
from src.data_preparation.utils import MetadataIndex
# %%
import pandas as pd
import polars as pl
from pathlib import Path
# %%

data_folder = Path("data/source_tables/ken_datasets")
src_dataset_path = Path(data_folder, "us-accidents")
dataset_path  = Path(src_dataset_path, "us-accidents.csv")
ds = RawDataset(dataset_path, "ken_datasets", "data/metadata")


# %%
data_folder = Path("data/yago3-dl/wordnet")
src_dataset_path = Path(data_folder, "")
dataset_path  = Path(src_dataset_path, "yago_seltab_wordnet_site.parquet")
ds = RawDataset(dataset_path, "yago3-dl", "data/metadata")


# %%
for dataset_path in data_folder.glob("**/*.parquet"):
    ds = RawDataset(dataset_path, "yago3-dl", "data/metadata")
    ds.save_metadata_to_json()
# %%

############# Prepare debug subset
data_folder = Path("data/yago3-dl/wordnet/subtabs/yago_seltab_wordnet_movie")
for dataset_path in data_folder.glob("**/*.parquet"):
    ds = RawDataset(dataset_path, "yago3-dl", "data/metadata/debug")
    ds.save_metadata_to_json()

# %%
metadata_index = MetadataIndex("data/metadata/debug")
metadata_index.save_index("debug_metadata_index.pickle")
# %%
