# %%
import datetime as dt
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skrub import (
    AggJoiner,
    GapEncoder,
    MinHashEncoder,
    MultiAggJoiner,
    TableVectorizer,
    tabular_learner,
)

from src.utils.indexing import load_query_result

CACHE_PATH = "results/cache"


# %%
def prepare_skmodel():
    inner_model = make_pipeline(
        TableVectorizer(
            high_cardinality=MinHashEncoder(),
            low_cardinality=OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            ),
            n_jobs=32,
        ),
        HistGradientBoostingRegressor(),
        # RandomForestRegressor(n_jobs=32),
        memory=CACHE_PATH,
    )

    return inner_model


def fit_predict_skmodel(X_train, X_valid, y_train, model):

    model.fit(X_train, y_train)

    return model.predict(X_valid)


def prepare_catboost(X):
    defaults = {
        "l2_leaf_reg": 0.01,
        "od_type": "Iter",
        "od_wait": 10,
        "iterations": 100,
        "verbose": 0,
    }

    cat_features = X.select(cs.string()).columns

    parameters = dict(defaults)
    parameters["random_seed"] = 42
    return CatBoostRegressor(cat_features=cat_features, **parameters)


def prepare_table_catboost(table):
    table = table.fill_null(value="null").fill_nan(value=np.nan)
    return table.to_pandas()


def fit_predict_catboost(X_train, X_valid, y_train, model: CatBoostRegressor):
    model.fit(X_train, y_train)

    return model.predict(X_valid)


# %%
def prepare():
    df = pl.read_parquet(
        "data/source_tables/yadl/company_employees-yadl-depleted.parquet"
    )

    X = df.drop("target")
    y = df["target"]

    query_info = {
        "data_lake": "wordnet_full",
        "table_path": "data/source_tables/yadl/company_employees-yadl-depleted.parquet",
        "query_column": "col_to_embed",
        "top_k": 10,
        "join_discovery_method": "exact_matching",
    }

    query_tab_path = Path(query_info["table_path"])
    if not query_tab_path.exists():
        raise FileNotFoundError(f"File {query_tab_path} not found.")

    tab_name = query_tab_path.stem
    query_result = load_query_result(
        query_info["data_lake"],
        query_info["join_discovery_method"],
        tab_name,
        query_info["query_column"],
        top_k=query_info["top_k"],
    )

    candidate_joins = query_result.candidates

    cjoin = next(iter(candidate_joins.values()))

    return X, y, cjoin


#%%
def merge(X_train, X_valid, cjoin):
    _, cnd_md, left_on, right_on = cjoin.get_join_information()

    cnd_table = pl.read_parquet(cnd_md["full_path"])

    aggr = AggJoiner(
        cnd_table, main_key=left_on, aux_key=right_on, operations=["mean", "mode"]
    )
    X_merged_train = aggr.fit_transform(X_train)
    X_merged_valid = aggr.fit_transform(X_valid)

    return X_merged_train, X_merged_valid


# %%
# X, y, cjoin = prepare()
df = pl.read_parquet("company_employees-yadl-depleted.parquet")

X = df.drop("target")
y = df["target"]

cjoin = pickle.load(open("cjoin.pickle", "rb"))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

X_merged_train, X_merged_valid = merge(X_train, X_valid, cjoin)
# %%
# %%timeit
def test_sklearn(X_merged_train, X_merged_valid, y_train, y_valid):
    print("preparing sklearn")
    inner_model = prepare_skmodel()
    print("fitting")
    y_predict = fit_predict_skmodel(
        X_merged_train, X_merged_valid, y_train, inner_model
    )
    r2 = r2_score(y_valid, y_predict)
    print(r2)


# %%
def test_catboost(X_merged_train, X_merged_valid, y_train, y_valid):
    print("preparing catboost")
    model = prepare_catboost(X_merged_train)
    print("fitting")
    _X_train, _X_valid = prepare_table_catboost(X_merged_train), prepare_table_catboost(
        X_merged_valid
    )
    _y = y_train.to_pandas()
    y_predict = fit_predict_catboost(_X_train, _X_valid, _y, model)

    r2 = r2_score(y_valid, y_predict)
    print(r2)


# %%
# %%timeit
if sys.argv[0] == "sk":
    test_sklearn(X_merged_train, X_merged_valid, y_train, y_valid)
elif sys.argv[0] == "cb":
    test_catboost(X_merged_train, X_merged_valid, y_train, y_valid)
else:
    raise ValueError
# %%

# %%
