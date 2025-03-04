#%%
import polars as pl
from pathlib import Path
import json

# %% Read and format experiments
exp_ids = [
    "0709",
    "0710",
    "0711",
    "0712",
    "0713",
    "0714",
    "0715",
    "0716",
    "0717",
    "0718",
    "0719",
    "0720",
    "0721",
    "0722",
]


def unpack_dict(log_dict):
    dd = dict()
    n_splits = log_dict["n_splits"]
    for k, v in log_dict["query_info"].items():
        dd[k] = [v] * n_splits

    if log_dict["task"] == "classification":
        dd["results"] = [x["f1"] for x in log_dict["results"]]
    else:
        dd["results"] = [x["r2"] for x in log_dict["results"]]
    dd["estimator"] = [x["estimator"] for x in log_dict["results"]]
    return {
        k: dd[k]
        for k in [
            "data_lake",
            "join_discovery_method",
            "estimator",
            "table_path",
            "query_column",
            "top_k",
            "results",
        ]
    }


def get_run_df(run_path):
    dfs = []

    scenario_id = sum(1 for _ in Path(run_path, "run_logs").iterdir())
    for _i in range(scenario_id):
        json_path = Path(run_path, "json", str(_i) + ".json")
        log_path = Path(run_path, "run_logs", str(_i) + ".log")
        log_json = json.load(open(json_path, "r"))

        top_k = log_json["top_k"]
        exp_task = log_json["task"]

        _df = pl.read_csv(log_path)
        _df = _df.with_columns(top_k=pl.lit(top_k).alias("top_k"))
        if exp_task == "regression":
            _df = _df.with_columns(
                pl.lit(0.0).alias("auc"),
                pl.lit(0.0).alias("f1score"),
                prediction_metric=pl.col("r2score"),
            )
        else:
            _df = _df.with_columns(
                pl.lit(0.0).alias("r2score"),
                pl.lit(0.0).alias("rmse"),
                prediction_metric=pl.col("f1score"),
            )

        dfs.append(_df)
    return pl.concat(dfs)


paths = []
for f in Path("results/logs").iterdir():
    if f.stem.split("-")[0] in exp_ids:
        paths.append(f)

all_dfs = []
for f in paths:
    x = get_run_df(f)
    all_dfs.append(x)
df = pl.concat(all_dfs)
df = df.with_columns(table=pl.col("base_table").str.split("-").list.first()).filter(
    (pl.col("table") != "us_accidents_large") & (pl.col("estimator") == "full_join")
)

df.write_csv("results/results_topk.csv")
# %%
