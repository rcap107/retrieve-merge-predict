# %%
# %cd ~/bench

from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from joblib import Parallel, delayed

# %%
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.data_structures.indices import CountVectorizerIndex
from src.utils.indexing import load_query_result


# %%
def get_values(list_values):
    s = "    ".join(list_values) + "    "
    return [s]


# %%
def prepare_table(table_path):
    table = pl.read_parquet(table_path)
    table_name = table_path.stem
    res = []
    sep = " " * 4
    for col in table.select(cs.string()).columns:
        values = sep.join([_ for _ in table[col].to_list() if _ is not None])
        values += " " * 4
        key = f"{table_name}__{col}"
        res.append((key, values))
    return res


# %% [markdown]
# ### Read the base table

# %%
base_path = Path("data/source_tables/")
tab_name = "us-accidents-yadl.parquet"
tab_path = Path(base_path, tab_name)
df = pl.read_parquet(tab_path)

# %% [markdown]
# ### Create the `CountVectorizer`

# %% [markdown]
# `token_pattern=r"(?u)(<[\S]+>)[ ]{4}"` is needed to avoid issues with spaces within the
# cell values. `binary` can be set to `True` because for Jaccard Containment the only
# thing that matters is having non-zero values (rather than the count).

# %%
cv = CountVectorizer(token_pattern=r"(?u)(<[\S]+>)[ ]{4}", binary=True)

# %%
values = get_values(df["col_to_embed"].to_list())
cv.fit(values)

# %%
len(cv.vocabulary_)

# %% [markdown]
# Testing the overlap with a random candidate table in the YADL data lake.
#
# I am taking a column in the table which I know can has some overlap with `col_to_embed` in the current table. Then, I transform the column.

# %%
df_cand = pl.read_parquet(
    "/home/soda/rcappuzz/work/benchmark-join-suggestions/data/wordnet_big/yagowordnet_wordnet_radio_station/radio_station_isLocatedIn_hasWebsite.parquet"
)
cand_values = get_values(df_cand["isLocatedIn"].to_list())

# %%
cv = CountVectorizer(token_pattern=r"(?u)(<[\S]+>)[ ]{4}", binary=True)
values = get_values(df["col_to_embed"].to_list())
cv.fit(values)
X = cv.transform(cand_values)

# %% [markdown]
# The sum of non-zero values is equivalent to the overlap between the tokens and
# the values in the column.

# %%
(X > 0).sum() / X.shape[1]

# %% [markdown]
# To double check, this is the Jaccard Containment, measured explicitly.

# %%
left_on = "col_to_embed"
right_on = "isLocatedIn"
unique_source = df[left_on].unique()
unique_cand = df_cand[right_on].unique()

s1 = set(unique_source.to_list())
s2 = set(unique_cand.to_list())
print(len(s1.intersection(s2)) / len(s1))


# %% [markdown]
# ## Testing transforming all tables

# %% [markdown]
# Here I am testing how to transform all tables (and their columns).

# %%
# rr = []
rr = Parallel(n_jobs=1, verbose=0)(
    delayed(prepare_table)(pth) for pth in base_path.glob("*.parquet")
)
res = [_1 for _ in rr for _1 in _]

# %%
X = cv.transform([_[1] for _ in res])

# %% [markdown]
# ## Sorting the results for querying

# %%
sum_res = X.sum(axis=1).ravel()
s_index = sum_res.argsort()
keys = np.array([_[0] for _ in res])
# np.flip(keys[s_index])

# %%
sum_res = np.array(X.sum(axis=1)).ravel()
s_index = np.flip(sum_res.argsort())
keys = np.array([_[0] for _ in res])
#%%
ranking = [_.split("__") for _ in keys[s_index]]

#%%
sum_res = np.array(X.sum(axis=1)).ravel()
s_index = np.flip(sum_res.argsort())
split_keys = [_.split("__") for _ in np.flip(keys[s_index])]
voc_size = X.shape[1]
ranking = [a + [b / voc_size] for a, b in zip(split_keys, sum_res[s_index]) if b > 0]


# %%
# base_path = Path("data/wordnet_big")

# %%
total = sum([1 for _ in base_path.glob("**/*.parquet")])
# %%
rr = Parallel(n_jobs=-1, verbose=0)(
    delayed(prepare_table)(pth)
    for pth in tqdm(base_path.glob("**/*.parquet"), total=total)
)
res = [_1 for _ in rr for _1 in _]

# %%
cvi = CountVectorizerIndex(
    data_lake_path=base_path,
    base_table=df,
    query_column="col_to_embed",
)
# %%
