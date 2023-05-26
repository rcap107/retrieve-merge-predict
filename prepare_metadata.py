#%% 
%load_ext autoreload
%autoreload 2
from src.utils.data_structures import RawDataset
from src.data_preparation.utils import MetadataIndex
# %%
import pandas as pd
import polars as pl
from pathlib import Path
# %%
def save_single_table(dataset_path, dataset_source, metadata_dest):
    ds = RawDataset(dataset_path, dataset_source, metadata_dest)
    ds.save_metadata_to_json()

#%%
ds_name = "us-accidents"
data_folder = Path("data/source_tables/ken_datasets")
src_dataset_path = Path(data_folder, ds_name)
dataset_path  = Path(src_dataset_path, ds_name + ".csv")

save_single_table(dataset_path, "ken_datasets", "data/metadata/sources")

#%%
ds_name = "movies-prepared"
data_folder = Path("data/source_tables/ken_datasets")
src_dataset_path = Path(data_folder, "the-movies-dataset")
dataset_path  = Path(src_dataset_path, ds_name + ".parquet")

save_single_table(dataset_path, "ken_datasets", "data/metadata/sources")

#%%
for case in ["wordnet", "binary", "seltab"]:
    data_folder = Path(f"data/yago3-dl/{case}")
    if data_folder.exists():
        for dataset_path in data_folder.glob("**/*.parquet"):
            ds = RawDataset(dataset_path, "yago3-dl", f"data/metadata/{case}")
            ds.save_metadata_to_json(f"data/metadata/{case}")
            ds.save_metadata_to_json("data/metadata/full")
    else:
        raise FileNotFoundError(f"Invalid path {data_folder}")

#%% 
# Prepare metadata indices
for case in ["wordnet", "binary", "seltab", "full"]:
    metadata_index = MetadataIndex(f"data/metadata/{case}")
    metadata_index.save_index(f"data/metadata/mdi/md_index_{case}.pickle")
    