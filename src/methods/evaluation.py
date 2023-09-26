"""Evaluation methods"""
import itertools
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from catboost import CatBoostError, CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
from tqdm import tqdm

from src.data_structures.loggers import RawLogger, RunLogger
from src.utils.models import get_model

logger_sh = logging.getLogger("pipeline")

import src.utils.joining as utils

# TODO: move this somewhere else
model_folder = Path("data/models")
os.makedirs(model_folder, exist_ok=True)


def prepare_table_for_evaluation(src_df):
    df = src_df.with_columns(
        src_df.with_columns(cs.string().fill_null("null"), cs.float().fill_null(np.nan))
    )
    # df = src_df.fill_null()
    df = utils.cast_features(df)
    return df


def prepare_X_y(
    src_df,
    target_column,
):
    y = src_df[target_column].to_pandas()
    df = src_df.drop(target_column).fill_null(value="null").fill_nan(value=np.nan)
    df = utils.cast_features(df)
    cat_features = df.select(cs.string()).columns
    X = df.to_pandas()

    return X, y, cat_features


def base_table(
    scenario_logger,
    splits,
    base_table,
    target_column="target",
    iterations=500,
    verbose=0,
    catboost_parameters=None,
):
    if catboost_parameters is None:
        catboost_parameters = {"l2_leaf_reg": 0.01, "od_type": None, "od_wait": None}

    additional_parameters = {"aggregation": "nojoin", "join_strategy": "nojoin"}
    run_logger = RunLogger(scenario_logger, additional_parameters)
    run_logger.start_time("run")

    r2_results = []

    for idx, (train_split, test_split) in enumerate(splits):
        raw_logger = RawLogger(scenario_logger, idx, additional_parameters)
        raw_logger.start_time("run")
        run_logger.start_time("run", cumulative=True)
        left_table_train = base_table[train_split]
        left_table_test = base_table[test_split]

        X, y, cat_features = prepare_X_y(left_table_train, target_column)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        raw_logger.start_time("train")
        run_logger.start_time("train", cumulative=True)

        model = CatBoostRegressor(
            cat_features=cat_features,
            iterations=iterations,
            l2_leaf_reg=0.01,
            verbose=verbose,
            od_type="Iter",
            od_wait=10,
        )
        model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
        raw_logger.results["best_iteration"] = model.best_iteration_
        raw_logger.results["tree_count"] = model.tree_count_

        raw_logger.end_time("train")
        run_logger.end_time("train")

        raw_logger.start_time("eval")
        run_logger.start_time("eval", cumulative=True)
        eval_data = left_table_test.fill_nan("null").fill_null("null")
        y_test = eval_data[target_column].cast(pl.Float64)
        eval_data = eval_data.drop(target_column).to_pandas()

        y_pred = model.predict(eval_data)

        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)

        raw_logger.end_time("eval")
        run_logger.end_time("eval")

        raw_logger.results["r2score"] = r2
        raw_logger.results["n_cols"] = len(left_table_train.schema)

        raw_logger.set_run_status("SUCCESS")
        raw_logger.end_time("run")
        run_logger.end_time("run")
        raw_logger.to_raw_log_file()

    run_logger.results["avg_r2"] = np.mean(r2_results)
    run_logger.results["std_r2"] = np.std(r2_results)
    run_logger.set_run_status("SUCCESS")

    run_logger.to_run_log_file()

    logger_sh.info("Base table R2 %.4f" % (run_logger.results["avg_r2"]))

    results = {
        "index": "base_table",
        "case": "base_table",
        "avg_r2": run_logger.results["avg_r2"],
        "std_r2": run_logger.results["std_r2"],
    }

    return results


def single_join(
    scenario_logger,
    splits,
    join_candidates,
    index_name,
    base_table,
    iterations=1000,
    join_strategy="left",
    aggregation="first",
    top_k=None,
    target_column="target",
    verbose=0,
):
    additional_parameters = {
        "candidate_table": "best_candidate",
        "index_name": index_name,
        "join_strategy": "single_join",
    }
    run_logger = RunLogger(scenario_logger, additional_parameters=additional_parameters)
    run_logger.start_time("run")

    best_candidate_hash = None
    best_candidate_r2 = -np.inf

    dict_r2_by_cand = {}
    avg_r2_by_cand = {}

    overall_results = []

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
            "join_strategy": "single_join",
        }
        cnd_table = pl.read_parquet(cnd_md["full_path"])
        dict_r2_by_cand[hash_] = []
        for idx, (train_split, test_split) in enumerate(splits):
            raw_logger = RawLogger(
                scenario_logger=scenario_logger,
                fold_id=idx,
                additional_parameters=cand_parameters,
            )
            raw_logger.start_time("run")
            run_logger.start_time("run", cumulative=True)

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

            raw_logger.start_time("join")
            run_logger.start_time("join", cumulative=True)
            merged = utils.execute_join_with_aggregation(
                left_table_train,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                how=join_strategy,
                aggregation=aggregation,
                suffix="_right",
            )
            raw_logger.end_time("join")
            run_logger.end_time("join", cumulative=True)

            raw_logger.start_time("train")
            run_logger.start_time("train", cumulative=True)
            X, y, cat_features = prepare_X_y(merged, target_column)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

            model = CatBoostRegressor(
                cat_features=cat_features,
                iterations=iterations,
                l2_leaf_reg=0.01,
                verbose=verbose,
                od_type="Iter",
                od_wait=10,
            )

            model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
            raw_logger.results["best_iteration"] = model.best_iteration_
            raw_logger.results["tree_count"] = model.tree_count_
            raw_logger.end_time("train")
            run_logger.end_time("train")

            raw_logger.start_time("eval_join")
            run_logger.start_time("eval_join", cumulative=True)

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
            X, y, cat_features = prepare_X_y(merged_test, target_column)
            raw_logger.end_time("eval_join")
            run_logger.end_time("eval_join", cumulative=True)

            raw_logger.start_time("eval")
            run_logger.start_time("eval", cumulative=True)

            # eval_data = merged_test.fill_nan("null").fill_null("null")
            eval_data = prepare_table_for_evaluation(merged)
            y_test = eval_data[target_column].cast(pl.Float64)
            eval_data = eval_data.drop(target_column).to_pandas()

            y_pred = model.predict(eval_data)

            raw_logger.end_time("eval")
            run_logger.end_time("eval")

            r2 = r2_score(y_test, y_pred)
            dict_r2_by_cand[hash_].append(r2)
            overall_results.append({"candidate": hash_, "r2": r2})

            raw_logger.results["r2score"] = r2
            raw_logger.results["n_cols"] = len(merged_test.schema)

            raw_logger.set_run_status("SUCCESS")
            raw_logger.end_time("run")
            run_logger.end_time("run")
            raw_logger.to_raw_log_file()

        avg_r2 = np.mean(dict_r2_by_cand[hash_])
        avg_r2_by_cand[hash_] = avg_r2

        if avg_r2 > best_candidate_r2:
            best_candidate_hash = hash_
            best_candidate_r2 = avg_r2

    df_ranking = pl.from_dicts(overall_results)
    df_ranking = (
        df_ranking.groupby("candidate")
        .agg(pl.mean("r2").alias("avg_r2"), pl.std("r2").alias("std_r2"))
        .sort("avg_r2", descending=True)
    )

    if top_k is not None:
        df_ranking = df_ranking.limit(top_k)

    best_results = dict_r2_by_cand[best_candidate_hash]

    run_logger.results["avg_r2"] = np.mean(best_results)
    run_logger.results["std_r2"] = np.std(best_results)
    run_logger.results["best_candidate_hash"] = best_candidate_hash

    run_logger.set_run_status("SUCCESS")
    run_logger.to_run_log_file()

    logger_sh.info(
        "Best candidate: %s R2 %.4f" % (best_candidate_hash, best_candidate_r2)
    )

    results = {
        "index": index_name,
        "case": "single_join",
        "best_candidate_hash": best_candidate_hash,
        "avg_r2": run_logger.results["avg_r2"],
        "std_r2": run_logger.results["std_r2"],
    }

    return results, df_ranking


def full_join(
    scenario_logger,
    splits,
    join_candidates,
    index_name,
    base_table,
    target_column="target",
    iterations=1000,
    verbose=0,
    aggregation="first",
    case="full",
):
    additional_parameters = {
        "candidate_table": case,
        "index_name": index_name,
        "aggregation": aggregation,
        "join_strategy": f"{case}_join",
    }
    run_logger = RunLogger(scenario_logger, additional_parameters=additional_parameters)

    if aggregation == "dfs":
        logger_sh.error("Full join not available with DFS.")
        run_logger.end_time("run")
        run_logger.set_run_status("FAILURE")
        run_logger.to_run_log_file()
        return {
            "index": index_name,
            "case": case,
            "best_candidate_hash": "",
            "avg_r2": np.nan,
            "std_r2": np.nan,
        }

    results = []
    tree_count_list = []

    for idx, (train_split, test_split) in enumerate(splits):
        raw_logger = RawLogger(scenario_logger, idx, additional_parameters)
        raw_logger.start_time("run")
        run_logger.start_time("run", cumulative=True)

        left_table_train = base_table[train_split]
        left_table_test = base_table[test_split]

        ##### START JOIN
        raw_logger.start_time("join")
        run_logger.start_time("join", cumulative=True)

        merged = left_table_train.clone().lazy()
        merged = utils.execute_join_all_candidates(merged, join_candidates, aggregation)
        merged = prepare_table_for_evaluation(merged)

        raw_logger.end_time("join")
        run_logger.end_time("join", cumulative=True)
        # END JOIN

        # START TRAIN
        raw_logger.start_time("train")
        run_logger.start_time("train", cumulative=True)

        X, y, cat_features = prepare_X_y(merged, target_column)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        model = CatBoostRegressor(
            cat_features=cat_features,
            iterations=iterations,
            l2_leaf_reg=0.01,
            verbose=verbose,
            od_type="Iter",
            od_wait=10,
        )

        model.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
        raw_logger.results["best_iteration"] = model.best_iteration_
        raw_logger.results["tree_count"] = model.tree_count_
        tree_count_list += model.tree_count_

        raw_logger.end_time("train")
        run_logger.end_time("train")
        # END TRAIN

        # START EVAL JOIN
        raw_logger.start_time("eval_join")
        run_logger.start_time("eval_join", cumulative=True)

        merged_test = utils.execute_join_all_candidates(
            left_table_test, join_candidates, aggregation
        )
        raw_logger.end_time("eval_join")
        run_logger.end_time("eval_join", cumulative=True)
        # END EVAL JOIN

        # START EVAL
        raw_logger.start_time("eval")
        run_logger.start_time("eval", cumulative=True)
        # eval_data = merged_test.fill_nan("null").fill_null("null")
        eval_data = prepare_table_for_evaluation(merged_test)
        y_test = eval_data[target_column].cast(pl.Float64)
        eval_data = eval_data.drop(target_column).to_pandas()
        run_logger.end_time("eval")
        raw_logger.end_time("eval")

        y_pred = model.predict(eval_data)

        r2 = r2_score(y_test, y_pred)

        results.append(r2)
        raw_logger.results["r2score"] = r2
        raw_logger.results["n_cols"] = len(merged_test.schema)

        raw_logger.set_run_status("SUCCESS")
        raw_logger.end_time("run")
        run_logger.end_time("run")

        raw_logger.to_raw_log_file()

    run_logger.results["avg_r2"] = np.mean(results)
    run_logger.results["std_r2"] = np.std(results)
    run_logger.results["best_candidate_hash"] = "full_join"
    run_logger.to_run_log_file()

    logger_sh.info("Best %s R2 %.4f" % (case, run_logger.results["avg_r2"]))

    results = {
        "index": index_name,
        "case": case,
        "best_candidate_hash": "full_join",
        "avg_r2": run_logger.results["avg_r2"],
        "std_r2": run_logger.results["std_r2"],
        "mdn_tree_count": np.median(tree_count_list),
    }

    return results


def greedy_join(
    scenario_logger,
    splits,
    join_candidates,
    index_name,
    base_table,
    target_column="target",
    iterations=1000,
    verbose=0,
    aggregation="first",
):
    raise NotImplementedError
    additional_parameters = {
        "candidate_table": "greedy_join",
        "index_name": index_name,
        "aggregation": aggregation,
        "join_strategy": f"greedy_join",
    }
    run_logger = RunLogger(scenario_logger, additional_parameters=additional_parameters)

    if aggregation == "dfs":
        logger_sh.error("Full join not available with DFS.")
        run_logger.end_time("run")
        run_logger.set_run_status("FAILURE")
        run_logger.to_run_log_file()
        return {
            "index": index_name,
            "case": "greedy_join",
            "best_candidate_hash": "",
            "avg_r2": np.nan,
            "std_r2": np.nan,
        }

    results = []

    for idx, (train_split, test_split) in enumerate(splits):
        raw_logger = RawLogger(scenario_logger, idx, additional_parameters)
        raw_logger.start_time("run")
        run_logger.start_time("run", cumulative=True)

        left_table_train = base_table[train_split]
        left_table_test = base_table[test_split]

        ##### START JOIN
        raw_logger.start_time("join")
        run_logger.start_time("join", cumulative=True)

        merged = left_table_train.clone().lazy()

        used_candidates = []

        for hash_, mdata in tqdm(
            join_candidates.items(),
            total=len(join_candidates),
            leave=False,
            desc="Training on candidates",
        ):
            src_md, cnd_md, left_on, right_on = mdata.get_join_information()
            current_candidates = used_candidates + [hash_]

            cand_parameters = {
                "candidate_table": hash_,
                "index_name": index_name,
                "left_on": left_on,
                "right_on": right_on,
                "join_strategy": "single_join",
            }
            cnd_table = pl.read_parquet(cnd_md["full_path"])
            raw_logger = RawLogger(
                scenario_logger=scenario_logger,
                fold_id=idx,
                additional_parameters=cand_parameters,
            )

            raw_logger.start_time("join")
            run_logger.start_time("join", cumulative=True)
            merged = utils.execute_join_with_aggregation(
                left_table_train,
                cnd_table,
                left_on=left_on,
                right_on=right_on,
                how=join_strategy,
                aggregation=aggregation,
                suffix="_right",
            )
            raw_logger.end_time("join")
            run_logger.end_time("join", cumulative=True)

            raw_logger.start_time("train")
            run_logger.start_time("train", cumulative=True)
            X, y, cat_features = prepare_X_y(merged, target_column)

            model = CatBoostRegressor(
                cat_features=cat_features,
                iterations=iterations,
                l2_leaf_reg=0.01,
                verbose=verbose,
            )

            model.fit(X=X, y=y)

            raw_logger.end_time("train")
            run_logger.end_time("train")

            # TODO: test the performance vs the previous

        raw_logger.end_time("join")
        run_logger.end_time("join", cumulative=True)
        # END JOIN

        merged = prepare_table_for_evaluation(merged)

        # START TRAIN
        raw_logger.start_time("train")
        run_logger.start_time("train", cumulative=True)

        X, y, cat_features = prepare_X_y(merged, target_column)

        model = CatBoostRegressor(
            cat_features=cat_features,
            iterations=iterations,
            l2_leaf_reg=0.01,
            verbose=verbose,
        )

        model.fit(X=X, y=y)
        raw_logger.end_time("train")
        run_logger.end_time("train")
        # END TRAIN

        # START EVAL JOIN
        raw_logger.start_time("eval_join")
        run_logger.start_time("eval_join", cumulative=True)

        merged_test = utils.execute_join_all_candidates(
            left_table_test, join_candidates, aggregation
        )
        raw_logger.end_time("eval_join")
        run_logger.end_time("eval_join", cumulative=True)
        # END EVAL JOIN

        # START EVAL
        raw_logger.start_time("eval")
        run_logger.start_time("eval", cumulative=True)
        # eval_data = merged_test.fill_nan("null").fill_null("null")
        eval_data = prepare_table_for_evaluation(merged_test)
        y_test = eval_data[target_column].cast(pl.Float64)
        eval_data = eval_data.drop(target_column).to_pandas()
        run_logger.end_time("eval")
        raw_logger.end_time("eval")

        y_pred = model.predict(eval_data)

        r2 = r2_score(y_test, y_pred)

        results.append(r2)
        raw_logger.results["r2score"] = r2
        raw_logger.results["n_cols"] = len(merged_test.schema)

        raw_logger.set_run_status("SUCCESS")
        raw_logger.end_time("run")
        run_logger.end_time("run")

        raw_logger.to_raw_log_file()

    run_logger.results["avg_r2"] = np.mean(results)
    run_logger.results["std_r2"] = np.std(results)
    run_logger.results["best_candidate_hash"] = "full_join"
    run_logger.to_run_log_file()

    logger_sh.info("Best %s R2 %.4f" % (case, run_logger.results["avg_r2"]))

    results = {
        "index": index_name,
        "case": case,
        "best_candidate_hash": "full_join",
        "avg_r2": run_logger.results["avg_r2"],
        "std_r2": run_logger.results["std_r2"],
    }

    return results
