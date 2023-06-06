"""Evaluation methods"""
# TODO: Fix imports
import hashlib
import logging
from copy import deepcopy

import featuretools as ft
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostError, CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    cross_validate,
    train_test_split,
)
from tqdm import tqdm
from woodwork.logical_types import Categorical, Double

from src.data_preparation.utils import cast_features
from src.table_integration.utils_joins import (
    execute_join,
    prepare_dfs_table,
    execute_join_complete,
)
from src.utils.data_structures import RunLogger, ScenarioLogger

import datetime as dt
from pathlib import Path
import git


repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha

# logger = logging.getLogger("main_pipeline")
# alt_logger = logging.getLogger("evaluation_method")
# alt_logger.setLevel(logging.DEBUG)

# log_format = "%(message)s"
# res_formatter = logging.Formatter(fmt=log_format)

# rfh = logging.FileHandler(filename=f"results/results_{repo_sha}.log")
# rfh.setFormatter(res_formatter)

# alt_logger.addHandler(rfh)

# TODO: move this somewhere else
model_folder = Path("data/models")


def measure_rmse(y_true, y_pred, squared=False):
    rmse = mean_squared_error(y_true, y_pred, squared=squared)
    return rmse


def measure_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2


def prepare_table_for_evaluation(df):
    df, num_features, cat_features = cast_features(df)
    df = df.fill_nan("null").fill_null("null")
    return (
        df,
        num_features,
        cat_features,
    )


def prepare_table_for_evaluation_old(df, num_features=None, cat_features=None):
    if num_features is None:
        df, num_features, cat_features = cast_features(df)
    else:
        for col in num_features:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
        cat_features = [col for col in df.columns if col not in num_features]

    df = df.fill_null("null")
    return df, num_features, cat_features


def evaluate_model_on_test_split(test_split, run_label):
    # TODO: add proper target column name rather than hardcoding "target"
    test_split = test_split.fill_nan("null").fill_null("null")
    y_test = test_split["target"].cast(pl.Float64)
    test_split = test_split.drop("target").to_pandas()
    model_name = Path(model_folder, run_label)
    model = CatBoostRegressor()
    model.load_model(model_name)
    y_pred = model.predict(test_split)

    rmse = measure_rmse(y_test, y_pred)
    r2 = measure_r2(y_test, y_pred)

    return (rmse, r2)


def run_on_table_cross_valid(
    src_df: pl.DataFrame,
    num_features,
    cat_features,
    scenario_logger,
    target_column="target",
    run_label=None,
    verbose=0,
    iterations=1000,
    n_splits=5,
    test_split=None,
    random_state=42,
    additional_parameters=None,
    additional_timestamps=None,
    cuda=False,
):
    # TODO: Better error chekcing
    if run_label is None:
        raise ValueError
    y = src_df[target_column].to_pandas()
    # df = src_df.drop(target_column).to_pandas()
    df = src_df.drop(target_column)
    # df = df.fillna("null")
    df = df.fill_null(value="null")
    df = df.fill_nan(value=np.nan)
    df = df.to_pandas()

    if cuda:
        model = CatBoostRegressor(
            cat_features=cat_features,
            iterations=iterations,
            task_type="GPU",
            #   devices="0"
        )
    else:
        model = CatBoostRegressor(
            cat_features=cat_features, iterations=iterations, l2_leaf_reg=0.01
        )

    results = cross_validate(
        model,
        X=df,
        y=y,
        scoring=("r2", "neg_root_mean_squared_error"),
        cv=n_splits,
        n_jobs=4,
        fit_params={"verbose": verbose},
        return_estimator=True,
    )

    # for fold in range(n_splits):
    #     run_logger = RunLogger(
    #         scenario_logger,
    #         fold_id=fold,
    #         additional_parameters=additional_parameters,
    #     )
    #     run_logger.durations["fit_time"] = results["fit_time"][fold]
    #     run_logger.durations["score_time"] = results["score_time"][fold]
    #     run_logger.results["rmse"] = results["test_neg_root_mean_squared_error"][fold]
    #     run_logger.results["r2score"] = results["test_r2"][fold]
    #     run_logger.set_run_status("SUCCESS")

        # alt_logger.info(run_logger.to_str())

    best_res = np.argmax(results["test_r2"])

    best_estimator = results["estimator"][best_res]
    best_estimator.save_model(Path(model_folder, run_label))

    return (max(results["test_r2"]), run_label), best_estimator


def execute_on_candidates(
    join_candidates,
    source_table,
    test_table,
    aggregation,
    scenario_logger,
    num_features,
    cat_features,
    verbose=1,
    iterations=1000,
    n_splits=5,
    join_strategy="left",
    cuda=False,
):
    durations = {}
    join_durations = []
    train_durations = []
    
    result_list = []
    for index_name, index_cand in join_candidates.items():
        for hash_, mdata in tqdm(index_cand.items(), total=len(index_cand)):
            # TODO reimplement this to have a single function that takes the candidate and executes the entire join elsewhere
            src_md = mdata.source_metadata
            cnd_md = mdata.candidate_metadata
            candidate_table = pl.read_parquet(cnd_md["full_path"])
            left_on = mdata.left_on
            right_on = mdata.right_on

            start_join = dt.datetime.now()
            merged = execute_join_complete(
                source_table,
                candidate_table,
                left_on=left_on,
                right_on=right_on,
                how=join_strategy,
                aggregation=aggregation,
            )
            
            # logger.debug("End joining.")

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
            end_join = dt.datetime.now()
            join_duration = (end_join - start_join).total_seconds()
            join_durations.append(join_duration)

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

            start_train = dt.datetime.now()
            best_score, best_model = run_on_table_cross_valid(
                merged,
                num_features,
                cat_features,
                scenario_logger=scenario_logger,
                verbose=verbose,
                run_label=hash_,
                n_splits=n_splits,
                iterations=iterations,
                additional_parameters=add_params,
                additional_timestamps={},
                cuda=cuda,
            )
            end_train = dt.datetime.now()
            train_duration = (end_train - start_train).total_seconds()
            train_durations.append(train_duration)

            result_list.append((best_score, best_model))

    result_list.sort(key=lambda x: x[0], reverse=True)

    hash_of_best_candidate = result_list[0][0][1]
    best_candidate_mdata = join_candidates["minhash"][hash_of_best_candidate]
    cnd_md = best_candidate_mdata.candidate_metadata
    candidate_table = pl.read_parquet(cnd_md["full_path"])
    left_on = best_candidate_mdata.left_on
    right_on = best_candidate_mdata.right_on


    start_eval_join = dt.datetime.now()
    merged_test = execute_join_complete(
        test_table,
        candidate_table,
        left_on=left_on,
        right_on=right_on,
        how=join_strategy,
        aggregation=aggregation,
    )
    end_eval_join = dt.datetime.now()
    eval_join_duration = (end_eval_join - start_eval_join).total_seconds()

    start_eval = dt.datetime.now()
    rmse, r2 = evaluate_model_on_test_split(merged_test, hash_of_best_candidate)
    end_eval = dt.datetime.now()
    eval_duration = (end_eval - start_eval).total_seconds()

    durations["avg_join"] = np.mean(join_durations)
    durations["avg_train"] = np.mean(train_durations)
    durations["eval_join"] = eval_join_duration
    durations["eval"] = eval_duration


    return (rmse, r2), durations


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
