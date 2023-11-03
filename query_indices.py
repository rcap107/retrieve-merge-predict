# %%
from pathlib import Path

import polars as pl
import toml

from src.data_structures.metadata import MetadataIndex, RawDataset
from src.utils.indexing import load_index, query_index

# %%
config = toml.load("config/join_discovery/minhash_open_data_clf.toml")

jd_method = config["join_discovery_method"]
data_lake_version = config["data_lake"]

query_cases = config["query_cases"]

metadata_dir = Path(f"data/metadata/{data_lake_version}")
metadata_index_path = Path(f"data/metadata/_mdi/md_index_{data_lake_version}.pickle")
index_dir = Path(f"data/metadata/_indices/{data_lake_version}")
# %%
if not metadata_index_path.exists():
    raise FileNotFoundError(f"Path to metadata index {metadata_index_path} is invalid.")
mdata_index = MetadataIndex(
    data_lake_variant=data_lake_version, index_path=metadata_index_path
)
# %%
print("Loading index...")
index = load_index(data_lake_version, jd_method)

# %%
print("Querying...")
for query_case in query_cases:
    query_tab_path = Path(query_case["table_path"])
    query_column = query_case["query_column"]
    query_index(index, query_tab_path, query_column, mdata_index)


# %%
