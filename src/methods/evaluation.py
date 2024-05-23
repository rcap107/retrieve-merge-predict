"""Evaluation methods"""
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
from memory_profiler import memory_usage
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from tqdm import tqdm

import src.utils.joining as ju
from src.data_structures.loggers import RunLogger, ScenarioLogger
from src.methods.join_selectors import (
    BestSingleJoin,
    FullJoin,
    HighestContainmentJoin,
    NoJoin,
    StepwiseGreedyJoin,
    TopKFullJoin,
)

logger_sh = logging.getLogger("pipeline")

# TODO: move this somewhere else
model_folder = Path("data/models")
os.makedirs(model_folder, exist_ok=True)


def prepare_splits(run_parameters: dict, base_table: pl.DataFrame, group_column: str):
    """Prepare the crossvalidation splits.

    Args:
        run_parameters (dict): Dictionary that contains the run parameters.
        base_table (pl.DataFrame): Base table to be used for training.
        group_column (str): Column that will used for joining and that will be used to
        separate splits with the GroupShuffle.

    Raises:
        ValueError: Raise value error if the provided split_kind is not an acceptable value.

    Returns:
        list: List of indices to be used to set the crossvalidation splits.
    """
    split_kind = run_parameters["split_kind"]
    n_splits = run_parameters["n_splits"]
    test_size = run_parameters["test_size"]
    if split_kind == "group_shuffle":
        groups = base_table.select(
            pl.col(group_column).cast(pl.Categorical).cast(pl.Int32).alias("group")
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


def prepare_X_y(src_df: pl.DataFrame, target_column: str, schema: dict = None):
    """Given a dataframe with a given target column and a schema, execute casting and return the dataframe
    as X and y.

    Args:
        src_df (pl.DataFrame): Dataframe to prepare.
        target_column (str): Name of the target column to pass as y.
        schema (dict, optional): Schema to be used for casting the tables. Defaults to None.

    Returns:
        pd.DataFrame, pd.Series: Dataframe and target series to be used in the training.
    """
    y = src_df[target_column].to_pandas()
    df = src_df.clone().cast(schema).fill_null(value="null").fill_nan(value=np.nan)
    if schema is None:
        df = ju.cast_features(df)
    else:
        df = df.cast(schema)
    X = df.drop(target_column).to_pandas()

    return X, y


def prepare_estimator(
    estimator_name: str,
    estimator_parameters: dict,
    join_candidates: list,
    join_parameters,
):
    """Prepare the estimator based on the given parameters.

    Args:
        estimator_name (str): Name of the estimator.
        estimator_parameters (dict): Parameters to be provided to the estimator.
        join_candidates (list): Candidates for the join operation.
        join_parameters (dict): Parameters to be used in the join operation.

    Returns:
        An estimator with the given type.
    """
    estimator_parameters.pop("active")
    if estimator_name == "no_join":
        return NoJoin(**estimator_parameters)

    estimator_parameters["join_parameters"] = join_parameters
    estimator_parameters.update({"candidate_joins": join_candidates})

    if estimator_name == "top_k_full_join":
        return TopKFullJoin(**estimator_parameters)
    if estimator_name == "highest_containment":
        return HighestContainmentJoin(**estimator_parameters)
    if estimator_name == "best_single_join":
        return BestSingleJoin(**estimator_parameters)
    if estimator_name == "full_join":
        return FullJoin(**estimator_parameters)
    if estimator_name == "stepwise_greedy_join":
        return StepwiseGreedyJoin(**estimator_parameters)


def evaluate_joins(
    scenario_logger: ScenarioLogger,
    base_table: pl.DataFrame,
    join_candidates,
    target_column: str = "target",
    group_column: str = "col_to_embed",
    estim_parameters: dict | None = None,
    join_parameters: dict | None = None,
    model_parameters: dict | None = None,
    run_parameters: dict | None = None,
):
    """Evaluate the join estimators on the given base table and join candidates. Potential additioanl parameters
    are provided.

    Args:
        scenario_logger (ScenarioLogger): ScenarioLogger object used to track the results of the runs.
        base_table (pl.DataFrame): Base table to evaluate.
        join_candidates (_type_): List of candidate joins that will be used for the estimators.
        target_column (str, optional): Target column that is used for training the model. Defaults to "target".
        group_column (str, optional): Column that will be used for joining. Defaults to "col_to_embed".
        estim_parameters (dict | None, optional): Additional parameters for the estimator. Defaults to None.
        join_parameters (dict | None, optional): Additional parameters for the join operation. Defaults to None.
        model_parameters (dict | None, optional): Additional parameters to be passed to the ML model. Defaults to None.
        run_parameters (dict | None, optional): Additional parameters relative to the run. Defaults to None.

    Raises:
        ValueError: Raise ValueError if the number of provided estimators is 0.
    """
    splits = prepare_splits(run_parameters, base_table, group_column)

    estim_common_parameters = {
        "scenario_logger": scenario_logger,
        "target_column": target_column,
        "model_parameters": model_parameters,
        "task": run_parameters["task"],
    }

    res_list = []
    add_info_dict = {}

    for fold_id, (train_split, test_split) in tqdm(
        enumerate(splits),
        total=len(splits),
        desc="CV progress: ",
        position=1,
        leave=False,
    ):
        base_table_train = base_table[train_split]
        base_table_test = base_table[test_split]

        schema = base_table_train.schema
        X_train, y_train = prepare_X_y(base_table_train, target_column, schema=schema)
        X_test, y_test = prepare_X_y(base_table_test, target_column, schema=schema)

        # Prepare the estimators using the provided parameters
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
                    )
                )

        if len(estimators) == 0:
            raise ValueError("No estimators were prepared. ")

        for estim in estimators:
            # Prepare the logger for this run
            run_logger = RunLogger(
                scenario_logger,
                additional_parameters=estim.get_estimator_parameters(),
                fold_id=fold_id,
            )
            run_logger.start_time("run")
            if estim is None:
                run_logger.set_run_status("FAILURE")
                run_logger.end_time("run")
                run_logger.to_run_log_file()
                continue

            estim.clean_timers()
            # Execute the fit operation
            run_logger.start_time("fit")
            mem_usage = memory_usage(
                (
                    estim.fit,
                    (
                        X_train,
                        y_train,
                    ),
                ),
                timestamps=True,
                max_iterations=1,
            )
            run_logger.end_time("fit")
            run_logger.mark_memory(mem_usage, "fit")

            # Execute the predict operation
            run_logger.start_time("predict")
            mem_usage, y_pred = memory_usage(
                (
                    estim.predict,
                    (X_test,),
                ),
                timestamps=True,
                max_iterations=1,
                retval=True,
            )
            run_logger.end_time("predict")
            run_logger.mark_memory(mem_usage, "predict")

            # Evaluate the results
            mem_usage, results = memory_usage(
                (
                    run_logger.measure_results,
                    (
                        y_test,
                        y_pred,
                    ),
                ),
                timestamps=True,
                max_iterations=1,
                retval=True,
            )
            run_logger.mark_memory(mem_usage, "test")
            curr_res = {"estimator": estim.name, **results}

            # Additional info includes best candidate join and relative info
            run_logger.set_additional_info(estim.get_additional_info())
            run_logger.update_durations(additional_durations=estim.get_durations())

            run_logger.end_time("run")
            run_logger.set_run_status("SUCCESS")
            run_logger.to_run_log_file()
            res_list.append(curr_res)
            add_info_dict[f"{fold_id}_{estim.name}"] = run_logger.additional_info

    scenario_logger.additional_info = add_info_dict
    scenario_logger.results = pl.from_dicts(res_list)
    scenario_logger.print_results()
