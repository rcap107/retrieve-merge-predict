#%%
import polars as pl

from src.data_structures.retrieval_methods import InverseIndex

config = {
    "metadata_dir": "data/metadata/binary_update",
    "file_path": None,
}
ri = InverseIndex(**config)

base_table = pl.read_parquet(
    "data/source_tables/yadl/us_elections-yadl-depleted.parquet"
)
query = base_table["col_to_embed"].to_list()


query_result = ri.query_index(query)

ri.save_index("data/metadata/_indices/binary_update")

print(query_result)

ri = InverseIndex(file_path="data/metadata/_indices/binary_update/reverse_index.pickle")
base_table = pl.read_parquet(
    "data/source_tables/yadl/us_elections-yadl-depleted.parquet"
)
query = base_table["col_to_embed"].to_list()
query_result = ri.query_index(query)
print(query_result)

print()
