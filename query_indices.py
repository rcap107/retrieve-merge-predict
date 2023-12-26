# %%
from pathlib import Path

import toml

from src.data_structures.join_discovery_methods import ExactMatchingIndex
from src.data_structures.metadata import MetadataIndex
from src.utils.indexing import DEFAULT_INDEX_DIR, load_index, query_index

# %%
config = toml.load("config/retrieval/query-wordnet_full_flat.toml")

jd_methods = config["join_discovery_method"]
data_lake_version = config["data_lake"]
exact_matching = config.get("exact_matching", False)

query_cases = config["query_cases"]

metadata_dir = Path(f"data/metadata/{data_lake_version}")
metadata_index_path = Path(f"data/metadata/_mdi/md_index_{data_lake_version}.pickle")
# %%
if not metadata_index_path.exists():
    raise FileNotFoundError(f"Path to metadata index {metadata_index_path} is invalid.")
mdata_index = MetadataIndex(
    data_lake_variant=data_lake_version, index_path=metadata_index_path
)
# %%
if "minhash" in jd_methods:
    minhash_index = load_index(
        {"join_discovery_method": "minhash", "data_lake": data_lake_version}
    )

# %%
for query_case in query_cases:
    tname = Path(query_case["table_path"]).stem
    query_tab_path = Path(query_case["table_path"])
    query_column = query_case["query_column"]
    for jd_method in jd_methods:
        if jd_method == "minhash":
            index = minhash_index
        elif jd_method == "exact_matching":
            index_path = Path(
                DEFAULT_INDEX_DIR,
                data_lake_version,
                f"em_index_{tname}_{query_column}.pickle",
            )
            index = ExactMatchingIndex(file_path=index_path)
        else:
            raise ValueError
        query_index(
            index, query_tab_path, query_column, mdata_index, exact_matching=True
        )
# %%
