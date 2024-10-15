import itertools
import logging
import os
from pathlib import Path

import git
import polars as pl

print(f"polars pool size: {pl.threadpool_size()}")
import polars.selectors as cs

import src.methods.evaluation as em
from src.data_structures.loggers import ScenarioLogger
from src.utils.constants import SUPPORTED_MODELS
from src.utils.indexing import load_query_result

repo = git.Repo(search_parent_directories=True)
repo_sha = repo.head.object.hexsha


def prepare_logger():
    logger = logging.getLogger("main")
    logger_scn = logging.getLogger("scn_logger")

    if not logger_scn.hasHandlers():
        fh = logging.FileHandler("results/logs/main_log.log")
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(message)s")
        fh.setFormatter(fh_formatter)

        logger_scn.addHandler(fh)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler_formatter = logging.Formatter("'%(asctime)s %(message)s'")
        console_handler.setFormatter(console_handler_formatter)

        logger.addHandler(console_handler)

    return logger, logger_scn


def prepare_dirtree():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("data/metadata/queries", exist_ok=True)


def convert_to_list(item: dict | list | str):
    if isinstance(item, dict):
        return {k: convert_to_list(v) for k, v in item.items()}
    if isinstance(item, list):
        return item
    return [item]


def get_comb(config_dict: dict):
    keys, values = zip(*config_dict.items())
    run_variants = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return run_variants


def flatten(key: str, this_dict: dict):
    flattened_dict = {}
    for k, v in this_dict.items():
        if key == "":
            this_key = f"{k}"
        else:
            this_key = f"{key}.{k}"
        if isinstance(v, list):
            flattened_dict.update({this_key: v})
        else:
            flattened_dict.update(flatten(this_key, v))
    return flattened_dict


def pack(dict_to_pack: dict):
    packed = {}
    for key, value in dict_to_pack.items():
        splits = key.split(".")
        n_splits = len(splits)
        if n_splits == 2:
            s0, s1 = splits
            if s0 not in packed:
                packed[s0] = {s1: value}
            else:
                packed[s0][s1] = value
        elif n_splits > 2:
            s0 = splits[0]
            pp = pack(
                {
                    k_.replace(s0 + ".", ""): v_
                    for k_, v_ in dict_to_pack.items()
                    if k_.startswith(s0)
                }
            )
            packed[s0] = pp
        elif n_splits == 1:
            packed[key] = value
        else:
            raise ValueError
    return packed


def prepare_config_dict(base_config: dict, debug=False):
    base_config["run_parameters"]["debug"] = debug
    converted_ = convert_to_list(base_config)

    flattened_ = flatten("", converted_)
    config_combinations = get_comb(flattened_)

    config_list = [pack(comb) for comb in config_combinations]

    for rv in config_list:
        validate_configuration(rv)

    return config_list


def validate_configuration(run_config: dict):
    run_parameters = run_config["run_parameters"]
    estim_parameters = run_config["estimators"]
    model_parameters = run_config["evaluation_models"]
    join_parameters = run_config["join_parameters"]
    query_info = run_config["query_cases"]

    # Check base table
    path_bt = Path(query_info["table_path"])
    tab_name = path_bt.stem
    print(f"Validating table {tab_name}")
    if not path_bt.exists():
        raise IOError(f"Base table file {path_bt} not found.")

    suffix = path_bt.suffix
    if not suffix in [".parquet", ".csv"]:
        raise ValueError(f"Extension {suffix} not supported.")

    if suffix == ".parquet":
        df = pl.read_parquet(path_bt)
    elif suffix == ".csv":
        df = pl.read_csv(path_bt)
    else:
        raise ValueError(f"Base table type {suffix} not supported.")

    if not query_info["query_column"] in df.columns:
        raise ValueError(
            f"Query column {query_info['query_column']} not in {df.columns}."
        )

    _tgt = run_parameters.get("target_column", "target")
    if not _tgt in df.columns:
        raise ValueError(f"Target column {_tgt} not in {df.columns}.")

    # Check run parameters
    assert run_parameters["task"] in [
        "regression",
        "classification",
    ], f"Task {run_parameters['task']} not supported"
    assert run_parameters["debug"] in [True, False]
    assert (
        isinstance(run_parameters["n_splits"], int) and run_parameters["n_splits"] > 0
    ), f"Incorrect value {run_parameters['n_splits']} for n_splits."
    assert (
        isinstance(run_parameters["test_size"], float)
        and 0 < run_parameters["test_size"] < 1
    )
    assert run_parameters["split_kind"] in ["group_shuffle"]

    # Check estimator parameters
    for estim, par in estim_parameters.items():
        if estim == "stepwise_greedy_join":
            assert par["budget_type"] in ["iterations"]
            assert isinstance(par["budget_amount"], int)
            assert isinstance(par["epsilon"], float) and 0 <= par["epsilon"] <= 1
            assert par["ranking_metric"] in ["containment"]
        assert par["active"] in [True, False]

    # Check model parameters
    assert model_parameters["chosen_model"] in SUPPORTED_MODELS
    if model_parameters["chosen_model"] == "catboost":
        par = model_parameters["catboost"]
        assert isinstance(par["iterations"], int) and par["iterations"] > 0
        assert par["od_type"] in ["Iter"]
        assert isinstance(par["od_wait"], int) and par["od_wait"] >= 0
        assert isinstance(par["l2_leaf_reg"], float) and par["l2_leaf_reg"] >= 0

    # Check join parameters
    assert join_parameters["join_strategy"] == "left"
    assert join_parameters["aggregation"] in ["dfs", "mean", "first"]

    # Check query parameters
    # TODO: fix this so it can be generalized
    # assert query_info["join_discovery_method"] in [
    #     "exact_matching",
    #     "minhash_hybrid",
    #     "minhash",
    # ]

    # Check query existence
    load_query_result(
        query_info["data_lake"],
        query_info["join_discovery_method"],
        tab_name,
        query_info["query_column"],
        validate=True,
    )


def single_run(run_config: dict, run_name=None):
    selector_parameters = run_config["estimators"]
    model_parameters = run_config["evaluation_models"]
    join_parameters = run_config["join_parameters"]
    run_parameters = run_config["run_parameters"]
    query_info = run_config["query_cases"]

    debug = run_parameters["debug"]

    prepare_dirtree()

    logger, logger_scn = prepare_logger()
    logger.info("Starting run.")

    query_tab_path = Path(query_info["table_path"])
    if not query_tab_path.exists():
        raise FileNotFoundError(f"File {query_tab_path} not found.")

    tab_name = query_tab_path.stem

    scl = ScenarioLogger(
        base_table_name=tab_name,
        git_hash=repo_sha,
        run_config=run_config,
        exp_name=run_name,
        debug=debug,
    )

    # try:
    query_result = load_query_result(
        query_info["data_lake"],
        query_info["join_discovery_method"],
        tab_name,
        query_info["query_column"],
        top_k=query_info["top_k"],
    )

    df_source = pl.read_parquet(query_tab_path).select(~cs.by_dtype(pl.Null)).unique()

    scl.add_timestamp("start_evaluation")
    logger.info("Starting evaluation.")
    em.evaluate_joins(
        scl,
        df_source,
        join_candidates=query_result.candidates,
        # TODO: generalize this
        target_column=run_parameters.get("target_column", "target"),
        group_column=query_info["query_column"],
        estim_parameters=selector_parameters,
        join_parameters=join_parameters,
        model_parameters=model_parameters,
        run_parameters=run_parameters,
    )
    scl.set_status("SUCCESS")
    # except Exception as exception:
    #     exception_name = exception.__class__.__name__
    #     scl.set_status("FAILURE", exception_name)

    logger.info("End evaluation.")
    scl.add_timestamp("end_evaluation")
    scl.add_timestamp("end_process")
    scl.add_process_time()

    scl.finish_run()
    # logger_scn.debug(scl.to_string())
    logger.info("Run end.")

    return scl
