# %%
from pathlib import Path

import polars as pl
from polars import selectors as cs

import src.pipeline as pipeline
from src._join_aggregator import JoinAggregator
from src.data_structures.loggers import ScenarioLogger
from src.data_structures.metadata import MetadataIndex, RawDataset
from src.utils.indexing import write_candidates_on_file
from src.methods.evaluation import perform_feature_selection

print("Imported")
# %%
def main():
    ##### Parameters
    yadl_version = "wordnet_big_num_cp"
    top_k = 10  # number of candidates to return
    selected_index = "minhash"
    query_tab_path = Path("data/source_tables/us-accidents-depleted.parquet")
    joinpath_file_name = "joinpaths.txt"
    query_column = "col_to_embed"

    # Prepare the metadata
    metadata_dir = Path(f"data/metadata/{yadl_version}")
    metadata_index_path = Path(f"data/metadata/_mdi/md_index_{yadl_version}.pickle")
    mdata_index = MetadataIndex(index_path=metadata_index_path)

    # Prepare the index
    index_dir = Path(f"data/metadata/_indices/{yadl_version}")
    index_path = Path(index_dir, selected_index + "_index.pickle")

    # Prepare the query table
    base_table = pl.read_parquet(query_tab_path)
    tab_name = query_tab_path.stem
    query_tab_metadata = RawDataset(
        query_tab_path.resolve(), "queries", "data/metadata/queries"
    )

    # Load index
    minhash_index = pipeline.load_index(index_path, tab_name)

    # Execute the query
    print("Querying")
    query_tab_metadata.save_metadata_to_json()
    query = base_table[query_column].drop_nulls()
    query_results = minhash_index.query_index(query)
    candidates = pipeline.generate_candidates(
        "minhash",
        query_results,
        mdata_index,
        query_tab_metadata.metadata,
        query_column,
        top_k,
    )

    # Write join paths on file
    write_candidates_on_file(candidates, joinpath_file_name)

    r_list = []
    print("Iterating")
    for hash_, candidate_join in candidates.items():
        src_md, cnd_md, left_on, right_on = candidate_join.get_join_information()
        src_df = pl.read_parquet(src_md["full_path"])
        cnd_df = pl.read_parquet(cnd_md["full_path"])
        cols_to_agg = [col for col in cnd_df.columns if col not in right_on]

        ja = JoinAggregator(tables=[(cnd_df, right_on, cols_to_agg)], main_key=left_on)
        y = src_df["target"].to_numpy()
        merged = ja.fit_transform(src_df, y=y)

        perform_feature_selection(merged, "target")

        break


#%%

if __name__ == "__main__":
    main()
