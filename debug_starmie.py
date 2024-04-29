#%%
# %load_ext autoreload
# %autoreload 2


#%%
from pathlib import Path

import toml

from src.data_structures.retrieval_methods import ExactMatchingIndex, StarmieWrapper

# %%


# %%
config = toml.load("config/retrieval/prepare/prepare-starmie-wordnet_10k.toml")
#%%
c1 = config["starmie"][0]
import_path = c1["import_path"]
#%%
# import_path = "/home/soda/rcappuzz/work/starmie/results/yadl/binary_update/query-results_cl_company_employees-yadl-depleted.parquet"
wr = StarmieWrapper(
    import_path=import_path, base_table_path="company_employees-yadl-depleted"
)

# %%
data_lake_version = "wordnet_vldb_3"
base_path = Path(
    f"/home/soda/rcappuzz/work/starmie/results/metadata/{data_lake_version}"
)

for pth in base_path.glob("*.parquet"):
    print(pth)
    base_table_path = Path(pth.stem.replace("query-results_cl_", "")).with_suffix(
        ".parquet"
    )
    output_dir = Path(f"data/metadata/_indices/{data_lake_version}")
    wr = StarmieWrapper(import_path=pth, base_table_path=base_table_path)
    print(wr.ranking.top_k(5, by="similarity"))
    wr.save_index(output_dir)

# %%
output_dir = Path(f"data/metadata/_indices/{data_lake_version}")

wr.save_index(output_dir)

# %%
