"""Evaluation methods"""
import hashlib
import logging
from copy import deepcopy

import featuretools as ft
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostError, CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from woodwork.logical_types import Categorical, Double

from src.data_preparation.utils import cast_features
from src.table_integration.utils_joins import execute_join, prepare_dfs_table
from src.utils.data_structures import RunLogger, ScenarioLogger

import datetime as dt
from pathlib import Path
import git

repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha

logger = logging.getLogger("main_pipeline")
alt_logger = logging.getLogger("evaluation_method")
alt_logger.setLevel(logging.DEBUG)

log_format = "%(message)s"
res_formatter = logging.Formatter(fmt=log_format)

rfh = logging.FileHandler(filename="results/results.log")
rfh.setFormatter(res_formatter)

alt_logger.addHandler(rfh)


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
    src_df: pl.DataFrame,
    num_features,
    cat_features,
    scenario_logger,
    target_column="target",
    verbose=0,
    iterations=1000,
    n_splits=5,
    additional_parameters=None,
    additional_timestamps=None
):
    y = src_df[target_column].to_pandas()
    # df = src_df.drop(target_column).to_pandas()
    df = src_df.drop(target_column)
    # df = df.fillna("null")
    df = df.fill_null(value="null")
    df = df.fill_nan(value=np.nan)
    df = df.to_pandas()

    k_fold = KFold(n_splits=n_splits)
    if len(df) < 5:
        raise ValueError

    for fold_id, (train_indices, test_indices) in enumerate(k_fold.split(df)):
        run_logger = RunLogger(
            scenario_logger,
            fold_id=fold_id,
            additional_parameters=additional_parameters,
        )
        run_logger.update_timestamps(additional_timestamps)
        
        run_logger.timestamps["run_start"] = dt.datetime.now()
        try:
            X_train = df.iloc[train_indices]
            y_train = y[train_indices]
            X_test = df.iloc[test_indices]
            y_test = y[test_indices]

            model = CatBoostRegressor(
                cat_features=cat_features,
                iterations=iterations,
                #   task_type="GPU",
                #   devices="0"
            )
            model.fit(X_train, y_train, verbose=verbose)
            y_pred = model.predict(X_test)
            rmse = measure_rmse(y_test, y_pred)
            r2score = measure_r2(y_test, y_pred)

            run_logger.results["rmse"] = rmse
            run_logger.results["r2score"] = r2score
            run_logger.timestamps["run_end"] = dt.datetime.now()
            run_logger.add_duration("run_start", "run_end", "run_duration")

            run_logger.set_run_status("SUCCESS")
        except (CatBoostError, ValueError) as exc:
            run_logger.timestamps["run_end"] = dt.datetime.now()
            run_logger.add_duration("run_start", "run_end")
            run_logger.set_run_status(f"FAILURE: {type(exc).__name__}")

        alt_logger.info(run_logger.to_str())
    return run_logger


def execute_on_candidates(
    join_candidates,
    source_table,
    scenario_logger,
    num_features,
    cat_features,
    verbose=1,
    iterations=1000,
    join_strategy="left",
    aggregation="none"
):
    result_dict = {}
    for index_name, index_cand in join_candidates.items():
        for hash_, mdata in tqdm(index_cand.items(), total=len(index_cand)):
            # TODO reimplement this to have a single function that takes the candidate and executes the entire join elsewhere
            src_md = mdata.source_metadata
            cnd_md = mdata.candidate_metadata
            source_table = pl.read_parquet(src_md["full_path"])
            candidate_table = pl.read_parquet(cnd_md["full_path"])
            left_on = mdata.left_on
            right_on = mdata.right_on

            add_timestamps = {
                "join_start": dt.datetime.now()
            }
            if aggregation == "dfs":
                logger.debug("Start DFS.")

                merged = prepare_dfs_table(
                    source_table,
                    candidate_table,
                    left_on=left_on,
                    right_on=right_on,
                )
                logger.debug("End DFS.")

            else:
                if aggregation == "dedup":
                    dedup = True
                    jstr = join_strategy + "_dedup"
                else:
                    jstr = join_strategy + "_none"
                    dedup = False
                logger.debug("Start joining.")
                merged = execute_join(
                    source_table,
                    candidate_table,
                    left_on=left_on,
                    right_on=right_on,
                    how=join_strategy,
                    dedup=dedup,
                )
                logger.debug("End joining.")

            add_timestamps["join_end"] = dt.datetime.now()
            # TODO: FIX NUM_ CAT_ FEATURES
            num_features = []
            cat_features = []
            for col in merged.columns:
                try:
                    merged.with_columns(pl.col(col).cast(pl.Float64))
                    num_features.append(col)
                except pl.ComputeError:
                    merged.with_columns(pl.col(col).cast(pl.Utf8))
                    cat_features.append(col)

            merged[cat_features].fill_null("null")
            merged[num_features].fill_nan(np.nan)

            add_params = {
                    "source_table": Path(src_md["full_path"]).stem,
                    "candidate_table": cnd_md["full_path"],
                    "index_name": index_name,
                    "left_on": left_on,
                    "right_on": right_on,
                    "similarity": mdata.similarity_score,
                    "size_prejoin": len(source_table),
                    "size_postjoin": len(merged),
            }

            run_on_table(
                merged,
                num_features,
                cat_features,
                scenario_logger=scenario_logger,
                verbose=verbose,
                iterations=iterations,
                additional_parameters=add_params,
                additional_timestamps=add_timestamps
            )
    return


def execute_full_join(
    join_candidates: dict,
    source_table: pl.DataFrame,
    source_metadata,
    scenario_logger,
    num_features,
    verbose,
    iterations,
):
    results_dict = {}
    for index_name, index_cand in join_candidates.items():
        merged = source_table.clone().lazy()
        hashes = []

        params = {
            "source_table": source_metadata["full_path"],
            "candidate_table": "full_join",
            "left_on": "full_join",
            "right_on": "full_join",
            "size_prejoin": len(source_table),
        }
        for hash_, mdata in tqdm(index_cand.items(), total=len(index_cand)):
            cnd_md = mdata.candidate_metadata
            hashes.append(cnd_md["hash"])
            candidate_table = pl.read_parquet(cnd_md["full_path"])

            left_on = mdata.left_on
            right_on = mdata.right_on
            merged = execute_join(
                source_table,
                candidate_table,
                left_on=left_on,
                right_on=right_on,
                how="left",
                dedup=True,
                suffix="_" + hash_[:10],
            )
        merged = merged.fill_null("")
        merged = merged.fill_nan("")

        md5 = hashlib.md5()
        md5.update(("".join(sorted(hashes))).encode())
        digest = md5.hexdigest()

        cat_features = [col for col in merged.columns if col not in num_features]
        merged = merged.fill_null("")
        merged = merged.fill_nan("")
        params["size_postjoin"] = len(merged)
        run_on_table(
            merged,
            num_features,
            cat_features,
            scenario_logger,
            verbose=verbose,
            iterations=iterations,
            additional_parameters=params,
        )

    return
