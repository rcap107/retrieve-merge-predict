"""Evaluation methods"""
# TODO: Fix imports
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from catboost import CatBoostError, CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from tqdm import tqdm
import src.utils.utils_joins as utils

# TODO: move this somewhere else
model_folder = Path("data/models")


def prepare_table_for_evaluation(df):
    df = utils.cast_features(df)
    df = df.fill_nan("null").fill_null("null")
    return df


def evaluate_model_on_test_split(test_split, run_label, target_column_name=None):
    if target_column_name is None:
        target_column_name = "target"

    test_split = test_split.fill_nan("null").fill_null("null")
    y_test = test_split[target_column_name].cast(pl.Float64)
    test_split = test_split.drop(target_column_name).to_pandas()
    model_name = Path(model_folder, run_label)
    model = CatBoostRegressor()
    model.load_model(model_name)
    y_pred = model.predict(test_split)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return (rmse, r2)


def run_on_table_cross_valid(
    src_df: pl.DataFrame,
    target_column="target",
    run_label=None,
    verbose=0,
    iterations=1000,
    n_splits=5,
    additional_parameters=None,
    additional_timestamps=None,
    cuda=False,
    n_jobs=1,
):
    y = src_df[target_column].to_pandas()
    df = src_df.drop(target_column)
    df = utils.cast_features(df)
    cat_features = df.select(cs.string()).columns
    df = df.fill_null(value="null").fill_nan(value=np.nan).to_pandas()

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
        n_jobs=n_jobs,
        fit_params={"verbose": verbose},
        return_estimator=True,
    )

    best_res = np.argmax(results["test_r2"])

    best_estimator = results["estimator"][best_res]
    best_estimator.save_model(Path(model_folder, run_label))

    return (max(results["test_r2"]), run_label), best_estimator


def execute_on_candidates(
    join_candidates,
    source_table,
    test_table,
    aggregation,
    best_candidates=1, 
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
            merged = utils.execute_join_with_aggregation(
                source_table,
                candidate_table,
                left_on=left_on,
                right_on=right_on,
                how=join_strategy,
                aggregation=aggregation,
            )

            merged = utils.cast_features(merged)
            num_features, cat_features = utils.get_cols_by_type(merged)

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
    merged_test = utils.execute_join_with_aggregation(
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

    durations["time_join"] = np.sum(join_durations)
    durations["time_train"] = np.sum(train_durations)
    durations["time_eval_join"] = eval_join_duration
    durations["time_eval"] = eval_duration

    return (rmse, r2), durations


def execute_full_join(
    join_candidates: dict,
    source_table: pl.DataFrame,
    test_table,
    source_metadata,
    aggregation,
    verbose,
    iterations,
):
    durations = {}
    join_durations = []
    train_durations = []

    for _, index_cand in join_candidates.items():
        merged = source_table.clone().lazy()

        params = {
            "source_table": source_metadata.path,
            "candidate_table": "full_join",
            "left_on": "full_join",
            "right_on": "full_join",
            "size_prejoin": len(source_table),
        }

        merged = utils.execute_join_all_candidates(merged, index_cand, aggregation)

        merged = merged.fill_null("")
        merged = merged.fill_nan("")

        run_on_table_cross_valid(
            merged,
            verbose=verbose,
            iterations=iterations,
            additional_parameters=params,
            run_label="full_join",
        )

        start_eval_join = dt.datetime.now()
        merged_test = utils.execute_join_all_candidates(
            test_table, index_cand, aggregation
        )
        end_eval_join = dt.datetime.now()
        eval_join_duration = (end_eval_join - start_eval_join).total_seconds()

        start_eval = dt.datetime.now()
        rmse, r2 = evaluate_model_on_test_split(merged_test, "full_join")
        end_eval = dt.datetime.now()
        eval_duration = (end_eval - start_eval).total_seconds()

        durations["time_join"] = np.sum(join_durations)
        durations["time_train"] = np.sum(train_durations)
        durations["time_eval_join"] = eval_join_duration
        durations["time_eval"] = eval_duration

    return (rmse, r2), durations
