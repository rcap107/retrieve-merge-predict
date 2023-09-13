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
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

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


def evaluate_model_on_test_split(test_split, model, target_column_name=None):
    if target_column_name is None:
        target_column_name = "target"

    test_split = test_split.fill_nan("null").fill_null("null")
    y_test = test_split[target_column_name].cast(pl.Float64)
    test_split = test_split.drop(target_column_name).to_pandas()
    y_pred = model.predict(test_split)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return (rmse, r2)


def evaluate_single_table(
    src_df: pl.DataFrame,
    target_column="target",
    group_column="col_to_embed",
    run_label=None,
    verbose=0,
    iterations=1000,
    n_splits=5,
    cuda=False,
    n_jobs=1,
    with_model_selection=True,
):
    y = src_df[target_column].to_pandas()
    df = src_df.drop(target_column)
    df = utils.cast_features(df)
    cat_features = df.select(cs.string()).columns
    groups = df.select(
        pl.col(group_column).cast(pl.Categorical).cast(pl.Int16).alias("group")
    ).to_numpy()

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
            cat_features=cat_features,
            iterations=iterations,
            l2_leaf_reg=0.01,
            verbose=verbose,
        )

    gkf = GroupKFold(n_splits)
    if with_model_selection:
        parameters = {}
        clf = GridSearchCV(model, parameters, cv=gkf, n_jobs=n_jobs)
        results = clf.fit(X=df, y=y, groups=groups)
        best_estimator = results.best_estimator_
        best_score = results.best_score_
        best_estimator.save_model(Path(model_folder, run_label))
        return (run_label, best_estimator, best_score)
    else:
        results = cross_validate(
            model,
            X=df,
            y=y,
            scoring=("r2", "neg_root_mean_squared_error"),
            cv=gkf,
            # cv=n_splits,
            groups=groups,
            n_jobs=n_jobs,
            return_estimator=True,
        )
        best_res = np.argmax(results["test_r2"])
        best_estimator = results["estimator"][best_res]
        best_estimator.save_model(Path(model_folder, run_label))
        return (run_label, best_estimator, max(results["test_r2"]))


def run_on_base_table(
    scenario_logger,
    fold,
    left_table_train,
    left_table_test,
    target_column="target",
    iterations=500,
    n_splits=5,
    n_jobs=1,
    verbose=0,
    cuda=False,
    with_model_selection=True,
):
    run_logger = RunLogger(scenario_logger, fold, {"aggregation": "nojoin"})
    run_logger.start_time("run")
    run_logger.start_time("train")
    # logger_sh.info("Fold %d: Start training on base table" % (fold + 1))

    base_result = evaluate_single_table(
        left_table_train,
        target_column=target_column,
        n_splits=n_splits,
        run_label="base_table",
        verbose=verbose,
        iterations=iterations,
        cuda=cuda,
        n_jobs=n_jobs,
        with_model_selection=with_model_selection,
    )
    run_logger.end_time("train")

    run_logger.start_time("eval")
    eval_results = evaluate_model_on_test_split(left_table_test, base_result[1])
    run_logger.results["rmse"], run_logger.results["r2score"] = eval_results
    run_logger.results["n_cols"] = len(left_table_train.schema)

    run_logger.end_time("eval")
    run_logger.set_run_status("SUCCESS")
    run_logger.end_time("run")
    print(f"Base table R2: {run_logger.results['r2score']:.4f}")
    # logger_sh.info("Fold %d: End training on base table" % (fold + 1))
    run_logger.to_run_log_file()

    return eval_results


def run_on_candidates(
    scenario_logger,
    fold,
    join_candidates,
    index_name,
    left_table_train,
    left_table_test,
    iterations=1000,
    n_splits=5,
    join_strategy="left",
    aggregation="first",
    top_k=None,
    n_jobs=1,
    target_column="target",
    verbose=0,
    cuda=False,
    feature_selection=False,
    fs_iterations=50,
    with_model_selection=True,
):
    add_params = {"candidate_table": "best_candidate", "index_name": index_name}
    run_logger = RunLogger(scenario_logger, fold, additional_parameters=add_params)
    run_logger.start_time("run")
    logger_sh.info("Fold %d: SINGLE JOINS" % (fold + 1))

    y_train = left_table_train[target_column].cast(pl.Float64).to_numpy()
    y_test = left_table_test[target_column].cast(pl.Float64).to_numpy()

    best_candidate_hash = None
    best_candidate_model = None
    best_candidate_r2 = -np.inf
    best_candidate_features = None

    cand_count = 0
    result_list = []
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
        cand_logger = RunLogger(scenario_logger, fold, cand_parameters)
        cnd_table = pl.read_parquet(cnd_md["full_path"])

        # Join source table with candidate
        run_logger.start_time("join", cumulative=True)
        cand_logger.start_time("join")

        # ja = JoinAggregator(
        #     tables=[
        #         (
        #             candidate_table,
        #             right_on,
        #             [col for col in candidate_table.columns if col not in left_on],
        #         )
        #     ],
        #     main_key="col_to_embed",
        #     agg_ops=["mean", "min", "max", "mode"],
        # )

        # merged = ja.fit_transform(left_table_train, y=y_train)

        merged = utils.execute_join_with_aggregation(
            left_table_train,
            cnd_table,
            left_on=left_on,
            right_on=right_on,
            how=join_strategy,
            aggregation=aggregation,
            suffix="_right",
        )
        merged = prepare_table_for_evaluation(merged)

        if feature_selection:
            selected_features = perform_feature_selection(
                merged, target_column, fs_iterations
            )
            merged = merged[selected_features]
        else:
            selected_features = merged.columns

        run_logger.end_time("join", cumulative=True)
        cand_logger.end_time("join")
        run_logger.start_time("train", cumulative=True)
        cand_logger.start_time("train")
        # Result has format (run_label, best_estimator, best_R2score)
        result = evaluate_single_table(
            merged,
            verbose=verbose,
            run_label=hash_,
            n_splits=n_splits,
            iterations=iterations,
            cuda=cuda,
            n_jobs=n_jobs,
        )
        run_logger.end_time("train", cumulative=True)
        cand_logger.end_time("train")
        result_list.append(result)

        if result[2] > best_candidate_r2:
            best_candidate_hash, best_candidate_model, best_candidate_r2 = result
            best_candidate_features = selected_features

        cand_logger.results["r2score"] = result[2]
        cand_logger.set_run_status("SUCCESS")

        cand_logger.to_candidate_log_file()
        cand_count += 1

    result_list.sort(key=lambda x: x[2], reverse=True)

    best_candidate_mdata = join_candidates[best_candidate_hash]
    src_md, cnd_md, left_on, right_on = best_candidate_mdata.get_join_information()
    best_cnd_table = pl.read_parquet(cnd_md["full_path"])

    run_logger.start_time("eval_join")

    # merged_test = ja.transform(left_table_test)

    merged_test = utils.execute_join_with_aggregation(
        left_table_test,
        best_cnd_table,
        left_on=left_on,
        right_on=right_on,
        how=join_strategy,
        aggregation=aggregation,
        suffix="_right",
    )

    merged_test = prepare_table_for_evaluation(merged_test)

    # print(f"Merged: {len(merged_test.columns)} features")
    # if feature_selection:
    #     merged_test = merged_test[best_candidate_features]
    #     print(f"Merged selected: {len(merged_test.columns)} features")

    run_logger.end_time("eval_join")
    run_logger.start_time("eval")
    results_best = evaluate_model_on_test_split(merged_test, best_candidate_model)
    run_logger.end_time("eval")
    run_logger.results["rmse"], run_logger.results["r2score"] = results_best
    run_logger.results["n_cols"] = len(merged_test.schema)

    run_logger.set_run_status("SUCCESS")
    run_logger.to_run_log_file()
    # logger_sh.info("Fold %d: End training on candidates" % (fold + 1))

    print(f"Best single candidate R2: {run_logger.results['r2score']:.4f}")

    if top_k is not None:
        if top_k < 0:
            raise ValueError("`top_k` must be positive.")
        return results_best, [r[0] for r in result_list[:top_k]]
    return results_best


def run_on_full_join(
    scenario_logger,
    fold,
    join_candidates,
    index_name,
    left_table_train,
    left_table_test,
    target_column="target",
    iterations=1000,
    verbose=0,
    aggregation="first",
    cuda=False,
    case="full",
    n_jobs=1,
    feature_selection=False,
    fs_iterations=50,
    with_model_selection=True,
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
    run_logger = RunLogger(scenario_logger, fold, additional_parameters=add_params)
    run_logger.start_time("run")
    if case == "full":
        logger_sh.info("Fold %d: FULL JOIN" % (fold + 1))
    else:
        logger_sh.info("Fold %d: SAMPLED FULL JOIN" % (fold + 1))

    if aggregation == "dfs":
        logger_sh.error("Fold %d: Full join not available with DFS." % (fold + 1))
        run_logger.end_time("run")
        run_logger.set_run_status("FAILURE")
        run_logger.to_run_log_file()
        return [np.nan, np.nan]

    run_logger.start_time("join")
    merged = left_table_train.clone().lazy()
    merged = utils.execute_join_all_candidates(merged, join_candidates, aggregation)
    merged = prepare_table_for_evaluation(merged)

    if feature_selection:
        selected_features = perform_feature_selection(
            merged, target_column, fs_iterations
        )
        merged = merged[selected_features]

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
        with_model_selection=with_model_selection,
    )
    run_logger.end_time("train")

    run_logger.start_time("eval_join")
    merged_test = utils.execute_join_all_candidates(
        left_table_test, join_candidates, aggregation
    )
    merged_test = prepare_table_for_evaluation(merged_test)
    run_logger.end_time("eval_join")

    if feature_selection:
        merged_test = merged_test[selected_features]

    run_logger.start_time("eval")
    result_model = result_train[1]
    results = evaluate_model_on_test_split(merged_test, result_model)
    run_logger.results["rmse"], run_logger.results["r2score"] = results
    run_logger.end_time("eval")
    run_logger.set_run_status("SUCCESS")
    run_logger.end_time("run")
    # logger_sh.info("Fold %d: End training on full join" % (fold + 1))

    print(f"Best {case} join R2: {run_logger.results['r2score']:.4f}")

    run_logger.to_run_log_file()
    return results


def perform_feature_selection(df: pl.DataFrame, target_column, iterations=20):
    X = df.drop(target_column).to_pandas()
    # X = df.to_pandas()
    cat_features = df.drop(target_column).select(cs.string()).columns

    y = df.select(pl.col(target_column)).to_numpy()

    model = CatBoostRegressor(
        cat_features=cat_features, iterations=iterations, l2_leaf_reg=0.01, verbose=0
    )

    selector = SelectFromModel(model)

    selector.fit(X, y)
    selected_features = list(X.columns[selector.get_support()]) + [target_column]
    return selected_features
