"""Evaluation methods"""
import polars as pl
from catboost import CatBoostRegressor, CatBoostError
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tqdm import tqdm
import numpy as np

from src.data_preparation.utils import cast_features
from src.table_integration.utils_joins import execute_join
from src.utils.data_structures import RunResult
import hashlib

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
    run_logger: RunResult,
    target_column="target",
    test_size=0.2,
    random_state=42,
    verbose=1,
    iterations=1000,
    
):
    y = src_df[target_column].to_pandas()
    df = src_df.drop(target_column).to_pandas()
    df = df.fillna("null")
    # df = df.fill_null(value="null")
    # df = df.fill_nan(value="null")
    # df = df.to_pandas()
    
    k_fold = KFold(n_splits=5)
    
    run_logger.add_value("parameters", "iterations", iterations)

    r2_list = []
    rmse_list = []
    try:
        run_logger.add_time("start_training")
        
        for train_indices, test_indices in k_fold.split(df):
            X_train = df.iloc[train_indices]
            y_train = y[train_indices]
            X_test = df.iloc[test_indices]
            y_test = y[test_indices]

            model = CatBoostRegressor(cat_features=cat_features, iterations=iterations)
            model.fit(X_train, y_train, verbose=verbose)
            y_pred = model.predict(X_test)
            rmse = measure_rmse(y_test, y_pred)
            r2score = measure_r2(y_test, y_pred)

            r2_list.append(r2score)
            rmse_list.append(rmse)

        run_logger.add_time("end_training")
        run_logger.add_duration("start_training", "end_training", "training_duration")
        run_logger.add_value("results", "avg_rmse", np.array(rmse_list).mean())
        run_logger.add_value("results", "error_rmse", np.array(rmse_list).std())
        run_logger.add_value("results", "avg_r2score", np.array(r2_list).mean())
        run_logger.add_value("results", "error_r2score", np.array(r2_list).std())
        run_logger.set_run_status("SUCCESS")
    except CatBoostError as e:
        run_logger.add_time("end_training")
        run_logger.add_duration("start_training", "end_training", "training_duration")
        run_logger.add_value("results", "avg_rmse", np.nan)
        run_logger.add_value("results", "error_rmse", np.nan)
        run_logger.add_value("results", "avg_r2score", np.nan)
        run_logger.add_value("results", "error_r2score", np.nan)
        run_logger.set_run_status(f"FAILURE: {type(e).__name__}")

        # try:
            # run_logger.add_time("start_training")
            # model = CatBoostRegressor(cat_features=cat_features, iterations=iterations)
            # model.fit(X_train, y_train, verbose=verbose)
            # run_logger.add_time("end_training")
            # run_logger.add_duration("start_training", "end_training", "training_duration")
            # y_pred = model.predict(X_test)
            # results = {"y_test": y_test, "y_pred": y_pred, "status": "SUCCESS"}
            # rmse = measure_rmse(results["y_test"], results["y_pred"])
            # r2score = measure_r2(results["y_test"], results["y_pred"])

            # run_logger.add_value("results", "rmse", rmse)
            # run_logger.add_value("results", "r2score", r2score)
            # # run_logger.add_value("status", "r2score", r2score)

            # run_logger.set_run_status("SUCCESS")
            
        # except Exception as e:
        #     print(f"Exception raised: {type(e).__name__}")
        #     run_logger.add_time("start_training")
        #     run_logger.add_time("end_training")
        #     run_logger.add_duration(label_duration="training_duration")

        #     run_logger.add_value("results", "rmse", rmse)
        #     run_logger.add_value("results", "r2score", r2score)
        #     run_logger.set_run_status(f"FAILED - {type(e).__name__}")

            


    return run_logger


def execute_on_candidates(
    join_candidates,
    source_table,
    num_features,
    cat_features,
    verbose=1,
    iterations=1000,
):
    result_dict = {}
    for index_name, index_cand in join_candidates.items():
        for hash_, mdata in tqdm(index_cand.items(), total=len(index_cand)):
            run_logger = RunResult()
            run_logger.add_value("parameters", "index_name", index_name)
            run_logger.add_value("parameters", "iterations", iterations)
            
            src_md = mdata.source_metadata
            cnd_md = mdata.candidate_metadata
            source_table = pl.read_parquet(src_md["full_path"])
            candidate_table = pl.read_parquet(cnd_md["full_path"])
            tqdm.write(cnd_md["full_path"])
            left_on = mdata.left_on
            right_on = mdata.right_on
                    
            run_logger.add_value("parameters", "source_table", src_md["full_path"])
            run_logger.add_value("parameters", "candidate_table", cnd_md["full_path"])
            run_logger.add_value("parameters", "left_on", left_on)
            run_logger.add_value("parameters", "right_on", right_on)
            
            merged = execute_join(
                source_table,
                candidate_table,
                left_on=left_on,
                right_on=right_on,
                how="left",
            )
            merged = merged.fill_null("null")
            merged = merged.fill_nan(np.nan)
            num_features = [col for col, col_type in merged.schema.items() if col_type==pl.Float64]
            cat_features = [col for col in merged.columns if col not in num_features]
            run_logger = run_on_table(
                merged, num_features, cat_features, run_logger, verbose=verbose, iterations=iterations
            )
            result_dict[hash_] = run_logger
    return result_dict


def execute_full_join(
    join_candidates: dict, source_table: pl.DataFrame, source_metadata, num_features, verbose, iterations
):
    results_dict = {}
    for index_name, index_cand in join_candidates.items():
        run_logger = RunResult()
        run_logger.add_value("parameters", "index_name", index_name)
        run_logger.add_value("parameters", "iterations", iterations)

        merged = source_table.clone().lazy()
        hashes = []    
        
        for hash_, mdata in tqdm(index_cand.items(), total=len(index_cand)):
            cnd_md = mdata.candidate_metadata
            hashes.append(cnd_md["hash"])
            candidate_table = pl.read_parquet(cnd_md["full_path"])

            left_on = mdata.left_on
            right_on = mdata.right_on
            merged = merged.join(
                candidate_table.lazy(),
                left_on=left_on,
                right_on=right_on,
                how="left",
                suffix=f"_{hash_[:5]}",
            )
        merged = merged.collect()
        merged = merged.fill_null("")
        merged = merged.fill_nan("")


        md5 = hashlib.md5()
        md5.update(("".join(sorted(hashes))).encode())
        digest = md5.hexdigest()
        
        cat_features = [col for col in merged.columns if col not in num_features]
        merged = merged.fill_null("")
        merged = merged.fill_nan("")
        run_logger = run_on_table(
            merged,
            num_features,
            cat_features,
            run_logger,
            verbose=verbose,
            iterations=iterations,
        )
        run_logger.add_value("parameters", "source_table", source_metadata.info["full_path"])
        run_logger.add_value("parameters", "candidate_table", "full_join")
        
        results_dict[digest] = run_logger
    

    
    return results_dict
