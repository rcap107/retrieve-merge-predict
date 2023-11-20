# %%
import argparse
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from joblib import Parallel, delayed
from tqdm import tqdm

from src.data_structures.metadata import MetadataIndex, RawDataset
from src.utils.indexing import save_single_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", action="store")

    args = parser.parse_args()
    return args


# %%
con = sqlite3.connect("data/metadata.db")
cur = con.cursor()
# %%
# args = parse_args()
# a = {
#     "case": "binary_update",
#     "data_folder": "data/yadl/binary_update",
# }


# args = SimpleNamespace(**a)
# prepare_metadata_from_case(args.data_folder)


data_folder = Path("data/yadl/binary_update")
case = data_folder.stem
# os.makedirs(f"data/metadata/{case}", exist_ok=True)
# os.makedirs("data/metadata/_mdi", exist_ok=True)

total_files = sum(1 for f in data_folder.glob("**/*.parquet"))

# %%
for dataset_path in tqdm(data_folder.glob("**/*.parquet"), total=total_files):
    print(dataset_path)
    ds = RawDataset(dataset_path, "binary_update")

    break

# %%
cur.execute(f"DROP TABLE {case}")
con.commit()
# %%
q_create_table = (
    f"CREATE TABLE {case}(hash TEXT PRIMARY KEY, table_name TEXT, table_full_path TEXT)"
)
cur.execute(q_create_table)
con.commit()
# %%
query = f"INSERT INTO {case} VALUES(?, ?, ?);"

# %%
def insert_single_table(dataset_path, dataset_source):
    ds = RawDataset(dataset_path, dataset_source)
    return ds.get_as_tuple()


# %%
def prepare_metadata_from_case(data_folder):
    # logger.info("Case %s", case)
    data_folder = Path(data_folder)
    case = data_folder.stem

    total_files = sum(1 for f in data_folder.glob("**/*.parquet"))

    r = Parallel(n_jobs=1, verbose=0)(
        delayed(insert_single_table)(dataset_path, "binary_update")
        for dataset_path in tqdm(data_folder.glob("**/*.parquet"), total=total_files)
    )

    return r


# %%
r = prepare_metadata_from_case(data_folder)

#%%
cur.executemany(query, r)
con.commit()
# %%
q_list = [
    ("7f66497874c023541eef5a44e620352c",),
    ("8a43c1bd0fc0f9dab39004fdd211c34d",),
    ("8c129d261ce6359dabeaca1f5b392d2e",),
    ("91258ecfb767792f6de9c60b2f278e53",),
    ("94523ed420ebe27dafac561930efdde9",),
]
# %%

with sqlite3.connect("metadata.db") as con:
    cur = con.cursor()
    q_query_by_hash = (
        f"SELECT * FROM binary_update WHERE hash IN ({','.join(['?'] * len(q_list))})"
    )
    res = cur.execute(q_query_by_hash, qq)
    fetch = res.fetchall()

# %%
