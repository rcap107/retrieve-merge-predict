"""Evaluation methods"""
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import RunLogger
from src.methods.join_estimators import (
    BestSingleJoin,
    FullJoin,
    HighestContainmentJoin,
    NoJoin,
    StepwiseGreedyJoin,
)

# from tqdm.contrib.telegram import tqdm


logger_sh = logging.getLogger("pipeline")


# TODO: move this somewhere else
model_folder = Path("data/models")
os.makedirs(model_folder, exist_ok=True)


def prepare_splits(run_parameters, base_table=None, group_column=None):
    split_kind = run_parameters["split_kind"]
    n_splits = run_parameters["n_splits"]
    test_size = run_parameters["test_size"]
    if split_kind == "group_shuffle":
        groups = base_table.select(
            pl.col(group_column).cast(pl.Categorical).cast(pl.Int16).alias("group")
        ).to_numpy()
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=None)
        splits = gss.split(base_table, groups=groups)
    elif split_kind == "shuffle":
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=None)
        splits = ss.split(base_table)
    else:
        raise ValueError(f"Inappropriate value {split_kind} for `split_kind`.")

    splits = list(splits)
    return splits


def prepare_X_y(src_df, target_column, schema=None):
    y = src_df[target_column].to_pandas()
    df = src_df.clone().cast(schema).fill_null(value="null").fill_nan(value=np.nan)
    if schema is None:
        df = ju.cast_features(df)
    else:
        df = df.cast(schema)
    X = df.drop(target_column).to_pandas()

    return X, y


def prepare_estimator(
    estimator_name,
    estimator_parameters,
    join_candidates,
    join_parameters,
    base_table_schema,
):
    estimator_parameters.pop("active")
    if estimator_name == "no_join":
        return NoJoin(**estimator_parameters)

    estimator_parameters["join_parameters"] = join_parameters
    estimator_parameters.update({"candidate_joins": join_candidates})

    if estimator_name == "highest_containment":
        return HighestContainmentJoin(**estimator_parameters)
    if estimator_name == "best_single_join":
        return BestSingleJoin(**estimator_parameters)
    if estimator_name == "full_join":
        return FullJoin(**estimator_parameters)
    if estimator_name == "stepwise_greedy_join":
        return StepwiseGreedyJoin(**estimator_parameters)


def evaluate_joins(
    scenario_logger,
    base_table: pl.DataFrame,
    join_candidates,
    target_column="target",
    group_column="col_to_embed",
    estim_parameters=None,
    join_parameters=None,
    model_parameters=None,
    run_parameters=None,
):
    splits = prepare_splits(run_parameters, base_table, group_column)

    estim_common_parameters = {
        "scenario_logger": scenario_logger,
        "target_column": target_column,
        "model_parameters": model_parameters,
        "task": run_parameters["task"],
    }

    estimators = []

    for estim in estim_parameters:
        params = dict(estim_common_parameters)
        params.update(estim_parameters.get(estim, {}))
        if estim_parameters[estim]["active"]:
            estimators.append(
                prepare_estimator(
                    estim,
                    params,
                    join_candidates,
                    join_parameters,
                    base_table_schema=base_table.schema,
                )
            )

    if len(estimators) == 0:
        raise ValueError("No estimators were prepared. ")

    res_list = []

    for idx, (train_split, test_split) in tqdm(
        enumerate(splits),
        total=len(splits),
        desc="CV progress: ",
        position=1,
        leave=True,
    ):
        base_table_train = base_table[train_split]
        base_table_test = base_table[test_split]

        schema = base_table_train.schema
        X_train, y_train = prepare_X_y(base_table_train, target_column, schema=schema)
        X_test, y_test = prepare_X_y(base_table_test, target_column, schema=schema)

        for estim in estimators:
            run_logger = RunLogger(
                scenario_logger, additional_parameters=estim.get_estimator_parameters()
            )
            run_logger.start_time("run")
            if estim is None:
                run_logger.set_run_status("FAILURE")
                run_logger.end_time("run")
                continue

            run_logger.start_time("fit")
            estim.fit(X_train, y_train)
            run_logger.end_time("fit")

            run_logger.start_time("predict")
            y_pred = estim.predict(X_test)
            run_logger.end_time("predict")

            results = run_logger.measure_results(y_test, y_pred)
            curr_res = {"estimator": estim.name, **results}

            # Additional info includes best candidate join and relative info
            run_logger.set_additional_info(estim.get_additional_info())

            run_logger.end_time("run")
            run_logger.set_run_status("SUCCESS")
            run_logger.to_run_log_file()
            res_list.append(curr_res)

    scenario_logger.results = pl.from_dicts(res_list)
    print(scenario_logger.results)
