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

CACHE_PATH = "results/cache"


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


def merge(X_train, X_valid, cand_info):
    path = cand_info["candidate_path"]
    left_on = cand_info["left_on"]
    right_on = cand_info["right_on"]

    cnd_table = pl.read_parquet(path)

    aggr = AggJoiner(
        cnd_table, main_key=left_on, aux_key=right_on, operations=["mean", "mode"]
    )
    X_merged_train = aggr.fit_transform(X_train)
    X_merged_valid = aggr.fit_transform(X_valid)

    return X_merged_train, X_merged_valid


# %%
def test_multi(X, y, candidates, model):
    assert model in ["catboost", "sklearn"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    for k, cand_info in candidates:
        X_merged_train, X_merged_valid = merge(X_train, X_valid, cand_info)
        if model == "sklearn":
            test_sklearn(X_merged_train, X_merged_valid, y_train, y_valid)
        elif model == "catboost":
            test_catboost(X_merged_train, X_merged_valid, y_train, y_valid)


# %%
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
df = pl.read_parquet("company_employees-yadl-depleted.parquet")

X = df.drop("target")
y = df["target"]

candidates = pickle.load(open("candidates.pickle", "rb"))

model = "sklearn"
test_multi(X, y, candidates, "sklearn")
# %%
