#%%
# %load_ext autoreload
# %autoreload 2
#%%
from pathlib import Path

# %%
from src.data_structures.join_discovery_methods import CountVectorizerIndex

# %%
data_lake_path = "data/metadata/wordnet_big"
base_table_path = "data/source_tables/company-employees-yadl.parquet"
query_column = "col_to_embed"
# %%
cvi = CountVectorizerIndex(data_lake_path, base_table_path, query_column, binary=True)

# %%
out_path = Path("data/metadata/_indices/wordnet_big/")
cvi.save_index(out_path)
# %%
from joblib import load

# %%
# dc = load("data/metadata/_indices/wordnet_big/cv_index.pickle")

# # %%
# cv_load = CountVectorizerIndex(file_path="data/metadata/_indices/wordnet_big/cv_index.pickle")
# %%
cvi.query_index()
# %%
