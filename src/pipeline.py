import itertools
import logging
import os
from pathlib import Path

import git
import polars as pl

import src.methods.evaluation as em
from src.data_structures.loggers import ScenarioLogger
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


def convert_to_list(item):
    if isinstance(item, dict):
        return {k: convert_to_list(v) for k, v in item.items()}
    elif isinstance(item, list):
        return item
    else:
        return [item]


def get_comb(config_dict):
    keys, values = zip(*config_dict.items())
    run_variants = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return run_variants


def flatten(key, this_dict):
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


def pack(dict_to_pack):
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


def prepare_config_dict(base_config):
    converted_ = convert_to_list(base_config)

    flattened_ = flatten("", converted_)
    config_combinations = get_comb(flattened_)

    config_list = [pack(comb) for comb in config_combinations]

    return config_list


def single_run(run_config, run_name=None):
    estim_parameters = run_config["estimators"]
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

    try:
        query_result = load_query_result(
            query_info["data_lake"],
            query_info["join_discovery_method"],
            tab_name,
            query_info["query_column"],
            top_k=query_info["top_k"],
        )

        df_source = pl.read_parquet(query_tab_path).unique()

        scl.add_timestamp("start_evaluation")
        logger.info("Starting evaluation.")
        em.evaluate_joins(
            scl,
            df_source,
            join_candidates=query_result.candidates,
            # TODO: generalize this
            target_column="target",
            group_column=query_info["query_column"],
            estim_parameters=estim_parameters,
            join_parameters=join_parameters,
            model_parameters=model_parameters,
            run_parameters=run_parameters,
        )
        scl.set_status("SUCCESS")
    except Exception as exception:
        # raise exception
        exception_name = exception.__class__.__name__
        scl.set_status("FAILURE", exception_name)

    logger.info("End evaluation.")
    scl.add_timestamp("end_evaluation")
    scl.add_timestamp("end_process")
    scl.add_process_time()

    scl.finish_run()
    logger_scn.debug(scl.to_string())
    logger.info("Run end.")
