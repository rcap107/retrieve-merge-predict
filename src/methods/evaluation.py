"""Evaluation methods"""
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from catboost import CatBoostError, CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, GroupKFold
from tqdm import tqdm

from src.data_structures.loggers import RunLogger

logger_sh = logging.getLogger("pipeline")

import src.utils.joining as utils

# TODO: move this somewhere else
model_folder = Path("data/models")
os.makedirs(model_folder, exist_ok=True)


def prepare_table_for_evaluation(df):
    df = df.fill_nan("null").fill_null("null")
    df = utils.cast_features(df)
    return df


def prepare_X_y(
    src_df,
    target_column,
):
    y = src_df[target_column].to_pandas()
    df = src_df.drop(target_column)
    df = utils.cast_features(df)
    cat_features = df.select(cs.string()).columns
    X = df.fill_null(value="null").fill_nan(value=np.nan).to_pandas()

    return X, y, cat_features


def run_on_base_table(
    scenario_logger,
    splits,
    base_table,
    target_column="target",
    iterations=500,
    n_splits=5,
    n_jobs=1,
    verbose=0,
    cuda=False,
):
    run_logger = RunLogger(scenario_logger, splits, {"aggregation": "nojoin"})
    run_logger.start_time("run")
    run_logger.start_time("train")

    r2_results = []

    for idx, (train_split, test_split) in enumerate(splits):
        left_table_train = base_table[train_split]
        left_table_test = base_table[test_split]

        X, y, cat_features = prepare_X_y(left_table_train, target_column)

        model = CatBoostRegressor(
            cat_features=cat_features,
            iterations=iterations,
            l2_leaf_reg=0.01,
            verbose=verbose,
        )
        model.fit(X=X, y=y)

        eval_data = left_table_test.fill_nan("null").fill_null("null")
        y_test = eval_data[target_column].cast(pl.Float64)
        eval_data = eval_data.drop(target_column).to_pandas()

        y_pred = model.predict(eval_data)

        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)

        # run_logger.results["rmse"], run_logger.results["r2score"] = eval_results
        # run_logger.results["n_cols"] = len(left_table_train.schema)

        # run_logger.end_time("eval")
        # run_logger.set_run_status("SUCCESS")
        # run_logger.end_time("run")
        # logger_sh.info(
        #     "Fold %d: Base table R2 %.4f" % (splits + 1, run_logger.results["r2score"])
        # )

        # run_logger.to_run_log_file()

    return r2_results


def run_on_candidates(
    scenario_logger,
    splits,
    join_candidates,
    index_name,
    base_table,
    iterations=1000,
    n_splits=5,
    join_strategy="left",
    aggregation="first",
    top_k=None,
    n_jobs=1,
    target_column="target",
    verbose=0,
    cuda=False,
):
    add_params = {"candidate_table": "best_candidate", "index_name": index_name}
    run_logger = RunLogger(scenario_logger, splits, additional_parameters=add_params)
    run_logger.start_time("run")

    best_candidate_hash = None
    best_candidate_r2 = -np.inf

    dict_r2_by_cand = {}
    avg_r2_by_cand = {}
    for hash_, mdata in tqdm(
        join_candidates.items(),
        total=len(join_candidates),
        leave=False,
        desc="Training on candidates",
    ):
        src_md, cnd_md, left_on, right_on = mdata.get_join_information()
        cand_parameters = {
            "candidate_table": hash_,
            "index_name": index_name,
            "left_on": left_on,
            "right_on": right_on,
        }
        cand_logger = RunLogger(scenario_logger, splits, cand_parameters)
        cnd_table = pl.read_parquet(cnd_md["full_path"])

        dict_r2_by_cand[hash_] = []
        for idx, (train_split, test_split) in enumerate(splits):
            run_logger.start_time("join", cumulative=True)
            cand_logger.start_time("join")

            left_table_train = base_table[train_split]
            left_table_test = base_table[test_split]

            if False:
                ja = JoinAggregator(
                    tables=[
                        (
                            candidate_table,
                            right_on,
                            [
                                col
                                for col in candidate_table.columns
                                if col not in left_on
                            ],
                        )
                    ],
                    main_key="col_to_embed",
                    agg_ops=["mean", "min", "max", "mode"],
                )

                merged = ja.fit_transform(left_table_train, y=y_train)

            merged = utils.execute_join_with_aggregation(
                left_table_train,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                how=join_strategy,
                aggregation=aggregation,
                suffix="_right",
            )

            # Join source table with candidate
            X, y, cat_features = prepare_X_y(merged, target_column)

            model = CatBoostRegressor(
                cat_features=cat_features,
                iterations=iterations,
                l2_leaf_reg=0.01,
                verbose=verbose,
            )

            model.fit(X=X, y=y)

            if False:
                merged_test = ja.transform(left_table_test)
            merged_test = utils.execute_join_with_aggregation(
                left_table_test,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                how=join_strategy,
                aggregation=aggregation,
                suffix="_right",
            )

            eval_data = merged_test.fill_nan("null").fill_null("null")
            y_test = eval_data[target_column].cast(pl.Float64)
            eval_data = eval_data.drop(target_column).to_pandas()

            y_pred = model.predict(eval_data)

            r2 = r2_score(y_test, y_pred)
            dict_r2_by_cand[hash_].append(r2)

        avg_r2 = np.mean(dict_r2_by_cand[hash_])
        avg_r2_by_cand[hash_] = avg_r2

        if avg_r2 > best_candidate_r2:
            best_candidate_hash = hash_
            best_candidate_r2 = avg_r2

    return best_candidate_hash, best_candidate_r2


def run_on_full_join(
    scenario_logger,
    splits,
    join_candidates,
    index_name,
    base_table,
    target_column="target",
    iterations=1000,
    verbose=0,
    aggregation="first",
    cuda=False,
    case="full",
    n_jobs=1,
):
    """Evaluate the performance obtained by joining all the candidates provided
    by the join discovery algorithm, with no supervision.

    Args:
        scenario_logger (ScenarioLogger): Logger containing information relative to the current run.
        fold (int): Id of the outer fold.
        join_candidates (dict): Dictionary containing the candidates queried by the join discovery methods.
        left_table_train (pl.DataFrame): Train split for the left (source) table.
        left_table_test (pl.DataFrame): Test split for the left (source) table.
        iterations (int, optional): Number of iterations to be used by Catboost. Defaults to 1000.
        verbose (int, optional): Verbosity of the training model. Defaults to 0.
        aggregation (str, optional): Aggregation method to be used, can be either `first`, `mean` or `dfs`. Defaults to "first".
    cuda (bool, optional): Whether or not to train on GPU. Defaults to False.
        n_jobs (int, optional): Number of CPUs to use when training. Defaults to 1.
    """

    add_params = {
        "candidate_table": case,
        "index_name": index_name,
        "aggregation": aggregation,
    }
    run_logger = RunLogger(scenario_logger, splits, additional_parameters=add_params)
    run_logger.start_time("run")
    if case == "full":
        logger_sh.info("Fold %d: FULL JOIN" % (splits + 1))
    else:
        logger_sh.info("Fold %d: SAMPLED FULL JOIN" % (splits + 1))

    if aggregation == "dfs":
        logger_sh.error("Fold %d: Full join not available with DFS." % (splits + 1))
        run_logger.end_time("run")
        run_logger.set_run_status("FAILURE")
        run_logger.to_run_log_file()
        return [np.nan, np.nan]

    run_logger.start_time("join")
    merged = left_table_train.clone().lazy()
    merged = utils.execute_join_all_candidates(merged, join_candidates, aggregation)
    merged = prepare_table_for_evaluation(merged)

    run_logger.results["n_cols"] = len(merged.schema)
    run_logger.end_time("join")

    run_logger.start_time("train")
    result_train = evaluate_single_table(
        merged,
        verbose=verbose,
        iterations=iterations,
        run_label="full_join",
        cuda=cuda,
        n_jobs=n_jobs,
    )
    run_logger.end_time("train")

    run_logger.start_time("eval_join")
    merged_test = utils.execute_join_all_candidates(
        left_table_test, join_candidates, aggregation
    )
    merged_test = prepare_table_for_evaluation(merged_test)
    run_logger.end_time("eval_join")

    run_logger.start_time("eval")
    result_model = result_train[1]
    results = evaluate_model_on_test_split(merged_test, result_model)
    run_logger.results["rmse"], run_logger.results["r2score"] = results
    run_logger.end_time("eval")
    run_logger.set_run_status("SUCCESS")
    run_logger.end_time("run")

    logger_sh.info(
        "Fold %d: Best %s R2 %.4f" % (splits + 1, case, run_logger.results["r2score"])
    )

    run_logger.to_run_log_file()
    return results
