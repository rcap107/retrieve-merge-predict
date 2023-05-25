"""Evaluation methods"""
from pathlib import Path

import polars as pl
from catboost import CatBoostError, CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data_preparation.utils import cast_features
from src.table_integration.utils_joins import execute_join


def measure_rmse(y_true, y_pred, squared=False):
    rmse = mean_squared_error(y_true, y_pred, squared=squared)
    return rmse


def measure_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2


def prepare_table_for_evaluation(df, num_features=None, cat_features=None):
    if num_features is None:
        df, num_features, cat_features = cast_features(df)
    else:
        for col in num_features:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
        cat_features = [col for col in df.columns if col not in num_features]

    df = df.drop_nulls()
    # df = df.fill_null("")
    return df, num_features, cat_features


def run_on_table(
    df,
    num_features,
    cat_features,
    target_column="target",
    test_size=0.2,
    random_state=42,
    verbose=1,
):
    y = df[target_column]
    df = df.drop(target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        df.to_pandas(), y.to_pandas(), test_size=test_size, random_state=random_state
    )

    try:
        model = CatBoostRegressor(cat_features=cat_features)
        model.fit(X_train, y_train, verbose=verbose)
        y_pred = model.predict(X_test)
        results = {"y_test": y_test, "y_pred": y_pred, "status": "SUCCESS"}
    except Exception as e:
        results = {
            "y_test": None,
            "y_pred": None,
            "status": f"FAILED WITH EXCEPTION: {type(e).__name__}",
        }
    return results


def execute_on_candidates(
    join_candidates, source_table, num_features, cat_features, verbose=1
):
    result_dict = {}
    for hash_, mdata in tqdm(join_candidates.items(), total=len(join_candidates)):
        source_metadata = mdata.source_metadata
        candidate_metadata = mdata.candidate_metadata
        source_table = pl.read_parquet(source_metadata["full_path"])
        candidate_table = pl.read_parquet(candidate_metadata["full_path"])
        tqdm.write(candidate_metadata["full_path"])
        left_on = mdata.left_on
        right_on = mdata.right_on
        merged = execute_join(
            source_table,
            candidate_table,
            left_on=left_on,
            right_on=right_on,
            how="left",
        )
        merged = merged.fill_null("")
        cat_features = [col for col in merged.columns if col not in num_features]
        results = run_on_table(merged, num_features, cat_features, verbose=verbose)
        if results["status"] == "SUCCESS":
            rmse = mean_squared_error(
                y_true=results["y_test"], y_pred=results["y_pred"], squared=False
            )
            r2 = r2_score(
                y_true=results["y_test"], y_pred=results["y_pred"]
            )
            result_dict[hash_] = (results["status"], rmse, r2)
        else:
            result_dict[hash_] = (results["status"], None, None)
    return result_dict


def execute_full_join(
    candidates: dict, source_table: pl.DataFrame, num_features, cat_features
):
    merged = source_table.clone()

    for hash_, mdata in tqdm(candidates.items(), total=len(candidates)):
        source_metadata = mdata.source_metadata
        candidate_metadata = mdata.candidate_metadata
        source_table = pl.read_parquet(source_metadata["full_path"])
        candidate_table = pl.read_parquet(candidate_metadata["full_path"])
        left_on = mdata.left_on
        right_on = mdata.right_on
        merged = merged.join(
            candidate_table,
            left_on=left_on,
            right_on=right_on,
            how="left",
            suffix=f"_{hash_[:5]}",
        )

    cat_features = [col for col in merged.columns if col not in num_features]
    merged = merged.fill_null("")

    return merged
