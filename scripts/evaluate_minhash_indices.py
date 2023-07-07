"""Loading minhash indices and testing for containment to study the accuracy.
"""

# %%
import json
import polars as pl
import seaborn as sns
from pathlib import Path
from src.data_structures.indices import MinHashIndex
from src.pipeline import generate_candidates

from src.data_structures.metadata import CandidateJoin, RawDataset, MetadataIndex
import src.methods.join_profiling as jp
from sklearn.metrics import mean_squared_error

# %%
index_dir = Path("data/metadata/_indices/testing_minhash")

query_dir = Path("data/metadata/queries")


# %%
index_dict = {}
for pth in index_dir.glob("**/*.pickle"):
    index_name = pth.stem
    index = MinHashIndex()
    index.load_index(pth)
    index_dict[index_name] = index


# %%
mdata_index = MetadataIndex(index_path="data/metadata/_mdi/md_index_full.pickle")

list_dict = []
# %%
query_results = {}
for tab_path in query_dir.glob("*.json"):
    print(tab_path)
    mdata = json.load(open(tab_path))
    df = pl.read_parquet(mdata["full_path"])
    query = df["col_to_embed"].sample(int(3000)).drop_nulls()
    query_metadata = RawDataset(mdata["full_path"], "queries", "data/metadata/queries")
    for index_name, index in index_dict.items():
        query_results[index_name] = index.query_index(query)
    candidates_by_index = {}

    for index, index_res in query_results.items():
        candidates = generate_candidates(
            index, index_res, mdata_index, query_metadata, "col_to_embed", 20
        )
        candidates_by_index[index] = candidates
    for index_name, candidates in candidates_by_index.items():
        for c, cand in candidates.items():
            src_md = cand.source_metadata
            cand_md = cand.candidate_metadata
            left_on = cand.left_on
            right_on = cand.right_on
            source_table = pl.read_parquet(src_md["full_path"])
            candidate_table = pl.read_parquet(cand_md["full_path"])
            containment = jp.measure_containment(
                source_table, candidate_table, left_on, right_on
            )
            c_dict = {
                "index_name": index_name,
                "source_table": src_md["full_path"],
                "candidate_table": cand_md["full_path"],
                "similarity": cand.similarity_score,
                "containment": containment,
            }
            list_dict.append(c_dict)


# %%
results = pl.from_dicts(list_dict)
# %%
for g, group in results.groupby(["source_table", "index_name"]):
    print(g)

    print(mean_squared_error(group["containment"], group["similarity"], squared=False))
# %%
sub_results = (
    results.with_columns(
        pl.col("source_table").apply(lambda x: Path(x).stem),
        pl.col("index_name").apply(lambda x: x.replace("minhash_", "")),
        pl.col("containment") * 100,
        pl.col("similarity"),
        (pl.col("containment") * 100 - pl.col("similarity")).alias("difference"),
    )
    .sort(by=["source_table", "difference"])
    .drop("candidate_table")
)
# %%
sns.boxplot(sub_results.to_pandas(), x="source_table", y="difference", hue="index_name")
# %%
