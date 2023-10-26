#%%
from pathlib import Path

import polars as pl
import toml

import src.pipeline as pipeline
from src.data_structures.metadata import MetadataIndex, RawDataset

#%%
config = toml.load("config/proto_config2.ini")

jd_method = config["join_discovery_method"]
data_lake_version = config["data_lake"]

query_cases = config["query_cases"]

metadata_dir = Path(f"data/metadata/{data_lake_version}")
metadata_index_path = Path(f"data/metadata/_mdi/md_index_{data_lake_version}.pickle")
index_dir = Path(f"data/metadata/_indices/{data_lake_version}")
#%%
mdata_index = MetadataIndex(index_path=metadata_index_path)

#%%
query_tab_path = Path(query_cases[0]["table_path"])
query_column = query_cases[0]["query_column"]
#%%
if not query_tab_path.exists():
    raise FileNotFoundError(f"File {query_tab_path} not found.")

tab_name = query_tab_path.stem


if not metadata_index_path.exists():
    raise FileNotFoundError(f"Path to metadata index {metadata_index_path} is invalid.")
mdata_index = MetadataIndex(index_path=metadata_index_path)

indices = pipeline.load_indices(
    index_dir, selected_indices=jd_method, tab_name=tab_name
)
#%%
# Query index
df = pl.read_parquet(query_tab_path)
query_tab_metadata = RawDataset(
    query_tab_path.resolve(), "queries", "data/metadata/queries"
)
# query_tab_metadata.save_metadata_to_json()

if query_column not in df.columns:
    raise pl.ColumnNotFoundError()
query = df[query_column].drop_nulls()

#%%
query_results, candidates_by_index = pipeline.querying(
    query_tab_metadata.metadata,
    query_column,
    query,
    indices,
    mdata_index,
    args.top_k,
)

query_result_path = Path("results/generated_candidates")
os.makedirs(query_result_path, exist_ok=True)
with open(Path(query_result_path, f"{tab_name}.pickle"), "wb") as fp:
    pickle.dump(candidates_by_index, fp)

# %%
