# %%
import json
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy import sparse
from sklearn.utils import murmurhash3_32


# %%
def prepare_coo_matrix(path, col_keys):
    with open(path, "r") as fp:
        mdata_dict = json.load(fp)
    table_path = mdata_dict["full_path"]
    # Selecting only string columns
    table = pl.read_parquet(table_path)
    if len(table) == 0:
        return None
    ds_hash = mdata_dict["hash"]
    table.columns = [f"{ds_hash}__{k}" for k in table.columns]

    data = []
    i_arr = []
    j_arr = []

    for col, key in col_keys.items():
        values, counts = table[col].drop_nulls().value_counts()
        i = np.array([murmurhash3_32(val, positive=True) for val in values])

        j = key * np.ones_like(i)

        data.append(counts)
        i_arr.append(i)
        j_arr.append(j)

    # coo_matrix = sparse.coo_matrix(
    #     (np.concatenate(data), (np.concatenate(i_arr), np.concatenate(j_arr)))
    # )

    return (
        col_keys,
        (np.concatenate(data), (np.concatenate(i_arr), np.concatenate(j_arr))),
    )
    # return (column_ids, coo_matrix)


# %%
pth = Path("data/metadata/wordnet_small")
schema_mapping = {}
tot_col = 0
for idx, p in enumerate(pth.glob("**/*.json")):
    with open(p, "r") as fp:
        mdata_dict = json.load(fp)
    table_path = mdata_dict["full_path"]
    ds_hash = mdata_dict["hash"]
    schema = pl.read_parquet_schema(table_path)
    schema = {f"{ds_hash}__{k}": v for k, v in schema.items() if v == pl.String}
    schema_mapping[p] = dict(
        zip(schema.keys(), range(tot_col, tot_col + len(schema.keys())))
    )
    tot_col += len(schema)


# %%
result = []
for idx, p in enumerate(pth.glob("**/*.json")):
    r = prepare_coo_matrix(p, schema_mapping[p])
    if r is not None:
        result.append(r)
# %%
result = list(zip(*result))
keys = {v: k for d in schema_mapping.values() for k, v in d.items()}
mats = result[1]
# %%
new_data = np.concatenate([m[0] for m in mats])
new_coord = np.concatenate([m[1] for m in mats], axis=1)
# %%
coo = sparse.coo_matrix((new_data, (new_coord[0, :], new_coord[1, :])), dtype=np.uint32)

# %%
csr = coo.tocsr()
# %%
table = pl.read_parquet(
    Path("data/source_tables/yadl/company_employees-yadl-depleted.parquet")
).select(cs.string())
query = table["col_to_embed"].unique().to_numpy()
q_in = [murmurhash3_32(_, positive=True) for _ in query]
# %%
intersection = list(set(coo.row).intersection(set(q_in)))
r = np.array(csr[intersection].sum(axis=0)).squeeze() / len(intersection)
res = [(keys[_], r[_]) for _ in np.flatnonzero(r)]
# %%
